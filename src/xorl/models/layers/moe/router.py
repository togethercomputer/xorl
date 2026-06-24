"""Token-choice top-k router for MoE layers."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


_SYNTHETIC_ROUTING_ENV = "XORL_MOE_SYNTHETIC_ROUTING"


def balanced_synthetic_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")

    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1) * top_k
    topk_offsets = torch.arange(top_k, device=device, dtype=torch.long).unsqueeze(0)
    selected_experts = (token_offsets + topk_offsets) % num_experts
    routing_weights = torch.full((num_tokens, top_k), 1.0 / top_k, device=device, dtype=dtype)
    return routing_weights, selected_experts


def _synthetic_routing_mode() -> str | None:
    mode = os.environ.get(_SYNTHETIC_ROUTING_ENV, "").strip().lower()
    if mode in {"", "0", "false", "no", "off"}:
        return None
    if mode in {"balanced", "round_robin"}:
        return "balanced"
    raise ValueError(f"{_SYNTHETIC_ROUTING_ENV} must be unset or 'balanced', got {mode!r}")


def _balanced_selected_experts(router_logits: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Deterministically spread routed slots evenly over all experts.

    This is for synthetic profiling/benchmarking only. It preserves the normal
    routing-weight gather from router scores, but replaces top-k/hash expert
    selection with a balanced round-robin assignment. Matches the selection
    pattern produced by :func:`balanced_synthetic_routing`.
    """
    num_tokens = router_logits.shape[0]
    token_offsets = torch.arange(num_tokens, device=router_logits.device, dtype=torch.long).unsqueeze(1) * top_k
    slot_offsets = torch.arange(top_k, device=router_logits.device, dtype=torch.long).unsqueeze(0)
    return (token_offsets + slot_offsets) % num_experts


class TopKRouter(nn.Module):
    """Top-K routing.

    Default path: ``softmax -> topk -> optional renormalization`` (legacy
    behavior preserved for existing callers).

    DeepSeek-V4 path (opt-in via ``scoring_func="sqrtsoftplus"``):
    ``sqrt(softplus(logits))`` then either

    - **noaux_tc** (``topk_method="noaux_tc"``, no aux loss): top-k indices
      are picked from ``scores + e_score_correction_bias`` but the routing
      weights themselves are gathered from the *unbiased* scores.
    - **hash routing** (``tid2eid`` + ``input_ids`` provided to ``forward``):
      top-k indices come from a frozen ``tid2eid: [vocab_size, top_k]`` table
      keyed by token id. Routing weights are still ``sqrtsoftplus(logits)``
      gathered at those indices. Used by DSv4's first ``num_hash_layers``.

    In both V4 paths the routing weights are renormalized over the top-k
    slice and then multiplied by ``routed_scaling_factor``.

    The module is *stateless*: it does **not** own the gate ``nn.Linear``
    nor the ``e_score_correction_bias`` / ``tid2eid`` parameters. Those
    live on the caller (``MoEBlock`` for the gate, the V4 model for bias
    and the hash table) and are supplied each forward.

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token.
        norm_topk_prob: Whether to renormalize top-k routing weights
            (``softmax`` path only — V4 paths always renormalize).
        scoring_func: ``"softmax"`` (default) or ``"sqrtsoftplus"``.
        topk_method: ``None`` (default) or ``"noaux_tc"`` (V4 bias-aware
            top-k). Only meaningful when ``scoring_func == "sqrtsoftplus"``.
        routed_scaling_factor: Optional multiplier applied to routing weights
            (V4 paths only — the legacy softmax path ignores this for
            backward compatibility).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
        scoring_func: str = "softmax",
        topk_method: str | None = None,
        routed_scaling_factor: float | None = None,
    ):
        super().__init__()
        if scoring_func not in ("softmax", "sqrtsoftplus"):
            raise ValueError(f"Unsupported scoring_func: {scoring_func!r}")
        if topk_method is not None and topk_method != "noaux_tc":
            raise ValueError(f"Unsupported topk_method: {topk_method!r}")
        if topk_method == "noaux_tc" and scoring_func != "sqrtsoftplus":
            raise ValueError("topk_method='noaux_tc' requires scoring_func='sqrtsoftplus'")
        # ``routed_scaling_factor`` is only consumed on the V4 paths
        # (``_forward_sqrtsoftplus``); reject silently-ignored configs upfront.
        if routed_scaling_factor is not None and scoring_func == "softmax":
            raise ValueError(
                "routed_scaling_factor is only used on the sqrtsoftplus path; "
                "the legacy softmax router ignores it. Set scoring_func="
                "'sqrtsoftplus' or drop routed_scaling_factor from the config."
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor
        self.synthetic_routing_mode = _synthetic_routing_mode()
        # ``tid2eid`` is a frozen post-load buffer — validate bounds once and
        # remember the storage object so the hot path doesn't pay device->host
        # sync per forward.
        self._validated_tid2eid_ptr: int | None = None

    def _validate_tid2eid_once(self, tid2eid: torch.Tensor) -> None:
        """Bound-check the ``tid2eid`` lookup table the first time we see it.

        A corrupt table would silently produce an OOB index into
        ``MoEExperts.gate_up_proj`` (crash or read garbage depending on
        backend). Run the check once on the buffer itself; subsequent calls
        with the same storage are no-ops. ``data_ptr()`` is the
        identity proxy — copying the buffer (e.g. dtype cast) creates a new
        storage and re-validates, which is the desired behavior.

        **DO NOT mutate ``tid2eid`` in place after load.** An in-place
        ``copy_`` / scatter / ``fill_`` keeps ``data_ptr()`` unchanged and
        would skip re-validation, leaving a corrupt table silently in use.
        A freed-and-reallocated buffer at the same address has the same
        problem. Replace the buffer with a fresh tensor instead — that
        bumps the storage pointer and triggers a new validation. The
        contract assumes ``tid2eid`` is frozen post-load (true in all
        current load paths).

        ``torch.all`` forces a device→host sync; gate it under
        ``torch.compiler.is_compiling`` so a future ``torch.compile`` pass
        doesn't poison the graph with the sync.
        """
        if torch.compiler.is_compiling():
            return
        ptr = tid2eid.data_ptr()
        if self._validated_tid2eid_ptr == ptr:
            return
        assert torch.all(tid2eid >= 0), "tid2eid has negative entries"
        assert torch.all(tid2eid < self.num_experts), (
            f"tid2eid has indices >= num_experts={self.num_experts}; max={tid2eid.max().item()}"
        )
        self._validated_tid2eid_ptr = ptr

    def forward(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
        *,
        expert_bias: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ):
        """Compute routing weights and expert selection.

        Args:
            router_logits: Raw logits ``(num_tokens, num_experts)``.
            input_dtype: Dtype to cast final routing weights to.
            expert_bias: Optional ``(num_experts,)`` bias added only for
                top-k selection (``topk_method="noaux_tc"`` requires this).
            tid2eid: Optional frozen ``(vocab_size, top_k)`` int lookup
                table for hash-routed layers. When provided, ``input_ids``
                must also be provided and ``top_k`` indices come from the
                table instead of a top-k over scores.
            input_ids: ``(num_tokens,)`` int tensor of token ids; required
                iff ``tid2eid`` is provided.

        Returns:
            ``(routing_weights[num_tokens, top_k], selected_experts[num_tokens, top_k])``.
        """
        if self.scoring_func == "softmax":
            return self._forward_softmax(router_logits, input_dtype)
        # sqrtsoftplus path
        return self._forward_sqrtsoftplus(
            router_logits, input_dtype, expert_bias=expert_bias, tid2eid=tid2eid, input_ids=input_ids
        )

    def _forward_softmax(self, router_logits: torch.Tensor, input_dtype: torch.dtype):
        if _synthetic_routing_mode() == "balanced":
            return balanced_synthetic_routing(
                router_logits.shape[0],
                self.num_experts,
                self.top_k,
                router_logits.device,
                input_dtype,
            )
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)
        return routing_weights, selected_experts

    def _forward_sqrtsoftplus(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
        *,
        expert_bias: torch.Tensor | None,
        tid2eid: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ):
        # ``scores`` are the unbiased per-expert scores used to *weight* the
        # selected experts. Bias / hash-table only affects *which* experts
        # are selected.
        scores = F.softplus(router_logits.float()).sqrt().type_as(router_logits)

        if _synthetic_routing_mode() == "balanced":
            selected_experts = _balanced_selected_experts(router_logits, self.num_experts, self.top_k)
        elif tid2eid is not None:
            assert input_ids is not None, "tid2eid hash routing requires input_ids"
            assert not tid2eid.requires_grad
            assert not input_ids.requires_grad
            assert tid2eid.shape[1] == self.top_k, (
                f"tid2eid second dim {tid2eid.shape[1]} must equal top_k={self.top_k}"
            )
            # Bounds are validated once per ``tid2eid`` buffer (frozen post-load)
            # rather than once per forward — the previous per-step
            # ``torch.all(...)`` check forced two device→host syncs in a hot path.
            self._validate_tid2eid_once(tid2eid)
            selected_experts = tid2eid[input_ids].to(torch.long)
            assert selected_experts.shape == (router_logits.shape[0], self.top_k)
        else:
            if self.topk_method == "noaux_tc":
                assert expert_bias is not None, "noaux_tc requires expert_bias"
                scores_for_routing = scores + expert_bias
            else:
                scores_for_routing = scores
            _, selected_experts = torch.topk(scores_for_routing, self.top_k, dim=-1)

        routing_weights = torch.gather(scores, dim=1, index=selected_experts)
        # V4 paths always renormalize.
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor is not None:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(input_dtype)
        return routing_weights, selected_experts

    @classmethod
    def from_config(cls, config):
        """Create from a model config (e.g. ``Qwen3MoeConfig``, ``DeepseekV4Config``).

        Reads optional V4 fields ``scoring_func`` / ``topk_method`` /
        ``routed_scaling_factor`` via ``getattr`` so legacy configs that
        don't define them get the default softmax behavior.
        """
        # DeepseekV4 uses ``num_experts_per_tok`` + ``n_routed_experts``;
        # other models use ``num_experts_per_tok`` + ``num_experts``. Use
        # ``is None`` rather than ``or``: a config that legitimately set
        # ``num_experts = 0`` would otherwise silently fall through to the V4
        # field.
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None:
            num_experts = config.n_routed_experts
        # ``"default"`` is the HF/upstream sentinel for "no special selection
        # method"; map it to ``None`` so the V4 dispatch in ``forward`` can
        # use a clean ``self.topk_method is None`` check.
        topk_method = getattr(config, "topk_method", None)
        if topk_method == "default":
            topk_method = None
        scoring_func = getattr(config, "scoring_func", "softmax")
        # ``routed_scaling_factor`` is V4-only; only read it when we're on the
        # V4 scoring path. The constructor raises otherwise.
        routed_scaling_factor = (
            getattr(config, "routed_scaling_factor", None) if scoring_func == "sqrtsoftplus" else None
        )
        return cls(
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            scoring_func=scoring_func,
            topk_method=topk_method,
            routed_scaling_factor=routed_scaling_factor,
        )
