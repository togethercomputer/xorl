"""Model-level admission for GLM-5.2 full-parameter block-FP8 training.

Full-param mode is ONE admitted configuration.  The v1 trainable-target
allowlist is exactly:

- dense-layer MLPs (fused gate/up + down full-param linears);
- routed-expert banks (full-param GKN masters);
- the MoE router gate weight (FP32 master + published BF16 view).

EVERYTHING else stays frozen and the admission enforces it by exhaustive
parameter walk: attention/MLA projections and the absorbed kv_b path, the DSA
indexer, shared experts, LM head and embeddings, all norms and biases, the
router correction bias, and every
FP8 byte cache.  Any parameter that is trainable after installation but not
an admitted master RAISES — a configuration that silently trains the indexer
or kv_b must be impossible, not discouraged.

The admission logs engagement exactly once per process, naming every trained
and frozen parameter group with counts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from xorl.models.transformers.glm5.exact_fullparam_experts import (
    Glm52FullParamBlockFP8RoutedExperts,
)
from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    Glm52ExactFullParamRouterWeight,
    Glm52ExactTP1BlockFP8FullParamLinear,
    Glm52FullParamDenseMLP,
)
from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan, MLPType
from xorl.models.transformers.glm5.native_fp8 import (
    Glm52NativeBlockFP8Experts,
    validate_glm52_native_fp8_config,
)
from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear


logger = logging.getLogger(__name__)

GLM52_FULLPARAM_ADMISSION_CONTRACT_VERSION = "glm52_fullparam_admission_v1"

# The complete v1 trainable master inventory (parameter name suffixes owned
# by admitted full-param components).
_TRAINABLE_MASTER_SUFFIXES = (
    "weight_master",
    "gate_up_weight_master",
    "down_weight_master",
)

_engagement_logged = False


@dataclass(frozen=True)
class Glm52FullParamAdmissionReport:
    contract_version: str
    dense_mlp_layers: tuple[int, ...]
    routed_expert_layers: tuple[int, ...]
    router_layers: tuple[int, ...]
    trained_parameter_count: int
    frozen_parameter_count: int
    trained_group_sizes: dict[str, int]
    frozen_group_sizes: dict[str, int]
    # Per-scope memory envelopes derived from the installed masters on this
    # rank (not estimated from config): FP32 master bytes per
    # trained group, and the resident AdamW envelope (master + FP32 grad +
    # exp_avg + exp_avg_sq = 4x master bytes).
    trained_master_bytes_by_group: dict[str, int]
    adamw_step_bytes_by_group: dict[str, int]
    # Frozen-trunk activation-backward admission: counts of
    # frozen modules on which the validated activation dgrad was enabled so
    # trainable-set gradients can traverse the frozen trunk.  Zero would
    # mean the model cannot backpropagate past its first frozen boundary.
    frozen_dgrad_linear_count: int = 0
    frozen_dgrad_bank_count: int = 0


class Glm52FullParamTopkRouter(nn.Module):
    """Drop-in router replacement: full-param gate weight, inherited bias.

    Keeps the ``e_score_correction_bias`` buffer frozen, while the checkpoint
    path ``weight`` is replaced
    by the FP32-master/BF16-view mechanism.  The forward is the full-param
    router component's pinned BI router GEMM; the surrounding MoE block's
    selection program is unchanged.
    """

    _glm52_exact_fullparam_component = True
    fsdp_requires_full_precision = True

    def __init__(self, config, *, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.full_param = Glm52ExactFullParamRouterWeight(
            int(config.n_routed_experts),
            int(config.hidden_size),
            device=device,
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(int(config.n_routed_experts), dtype=torch.float32, device=device),
        )

    @classmethod
    def from_router(cls, router: nn.Module, config) -> "Glm52FullParamTopkRouter":
        weight = router.weight
        if weight.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 full-param router adoption requires a BF16 gate weight, got {weight.dtype}")
        bias = router.e_score_correction_bias
        if bias.dtype is not torch.float32:
            raise TypeError("GLM-5.2 e_score_correction_bias must remain FP32")
        replacement = cls(config, device=weight.device)
        replacement.full_param.load_from_bf16(weight.detach())
        with torch.no_grad():
            replacement.e_score_correction_bias.copy_(bias.detach())
        return replacement

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.full_param(hidden_states.reshape(-1, self.hidden_size))


def _dense_projection_bytes(module: nn.Module, name: str):
    if not isinstance(module, NativeBlockFP8Linear):
        raise RuntimeError(
            f"GLM-5.2 full-param admission requires {name} to be a frozen native block-FP8 "
            f"projection before installation, got {type(module).__name__}"
        )
    return module.fp8_weight(), module.weight_scale_inv.detach()


def install_glm52_fullparam_components(
    model: nn.Module,
    config,
    *,
    trainable_expert_layers: Sequence[int] | None = None,
    _skip_geometry_validation: bool = False,
) -> Glm52FullParamAdmissionReport:
    """Install full-param components per the layer plan and enforce the allowlist.

    ``trainable_expert_layers`` scopes expert-bank training to fit the
    available memory.  Only the named sparse layers receive full-param expert banks;
    every other sparse layer keeps its frozen native bank publishing
    step-0-identical bytes.  Dense MLPs and ALL routers are trained
    regardless of the expert scope.  ``None`` means every sparse layer for
    reduced structural tests; production must pass an explicit
    scope through :func:`prepare_glm52_fullparam_training`.

    ``_skip_geometry_validation`` exists ONLY for structural unit tests at
    reduced geometry (mirroring the exact-contract test convention); the
    production entry point is :func:`prepare_glm52_fullparam_training`, which
    never skips it.
    """

    if not _skip_geometry_validation:
        _validate_official_fullparam_config(config)
    validate_glm52_native_fp8_config(getattr(config, "quantization_config", None))
    if getattr(config, "hidden_act", None) != "silu":
        raise ValueError("GLM-5.2 full-param admission requires hidden_act='silu'")

    plan = Glm52LayerPlan.from_config(config)
    expert_scope = _validate_expert_scope(trainable_expert_layers, plan)
    layers = model.get_submodule("model.layers")

    dense_layers: list[int] = []
    expert_layers: list[int] = []
    router_layers: list[int] = []
    for spec in plan.layers:
        layer = layers[spec.layer_index]
        if spec.mlp_type is MLPType.DENSE:
            mlp = layer.mlp
            if isinstance(mlp, Glm52FullParamDenseMLP):
                raise RuntimeError(
                    f"layers.{spec.layer_index}.mlp is already a full-param dense MLP; admission is not idempotent"
                )
            gate_w, gate_s = _dense_projection_bytes(
                getattr(mlp, "gate_proj", None), f"layers.{spec.layer_index}.mlp.gate_proj"
            )
            up_w, up_s = _dense_projection_bytes(
                getattr(mlp, "up_proj", None), f"layers.{spec.layer_index}.mlp.up_proj"
            )
            down_w, down_s = _dense_projection_bytes(
                getattr(mlp, "down_proj", None), f"layers.{spec.layer_index}.mlp.down_proj"
            )
            replacement = Glm52FullParamDenseMLP(
                int(config.hidden_size),
                int(config.intermediate_size),
                device=gate_w.device,
            )
            replacement.load_prequantized_split(gate_w, gate_s, up_w, up_s, down_w, down_s)
            layer.mlp = replacement
            dense_layers.append(spec.layer_index)
            continue

        moe = layer.mlp
        bank = moe.experts
        if isinstance(bank, Glm52FullParamBlockFP8RoutedExperts):
            raise RuntimeError(
                f"layers.{spec.layer_index}.mlp.experts is already a full-param bank; admission is not idempotent"
            )
        if not isinstance(bank, Glm52NativeBlockFP8Experts):
            raise RuntimeError(
                f"GLM-5.2 full-param admission requires layers.{spec.layer_index}.mlp.experts to be the "
                f"frozen native block-FP8 bank, got {type(bank).__name__}"
            )
        if expert_scope is None or spec.layer_index in expert_scope:
            replacement_bank = Glm52FullParamBlockFP8RoutedExperts(
                int(bank.gate_up_packed_weight_f32.shape[0]),
                int(bank.hidden_size),
                int(bank.intermediate_size),
                device=bank.gate_up_packed_weight_f32.device,
            )
            replacement_bank.load_prequantized(
                bank.gate_up_proj,
                bank.gate_up_weight_scale_inv.detach(),
                bank.down_proj,
                bank.down_weight_scale_inv.detach(),
            )
            replacement_bank.assign_global_expert_range(
                _ep_expert_start(int(replacement_bank.num_experts), int(config.n_routed_experts)),
                int(config.n_routed_experts),
            )
            moe.experts = replacement_bank
            expert_layers.append(spec.layer_index)
        # Out-of-scope sparse layers keep the frozen native bank: it serves
        # and publishes step-0-identical bytes (the memory wall keeps its
        # masters off this rank), while its ROUTER is still trained below.

        moe.gate = Glm52FullParamTopkRouter.from_router(moe.gate, config)
        router_layers.append(spec.layer_index)

    # Freeze everything that is not an admitted master, then enforce the
    # allowlist by exhaustive walk.
    trained_groups: dict[str, int] = {"dense_mlp_masters": 0, "routed_expert_masters": 0, "router_masters": 0}
    trained_bytes: dict[str, int] = dict.fromkeys(trained_groups, 0)
    frozen_groups: dict[str, int] = {}
    trained = frozen = 0
    for name, parameter in model.named_parameters():
        is_master = name.endswith(_TRAINABLE_MASTER_SUFFIXES)
        if is_master:
            owner = model.get_submodule(name.rsplit(".", 1)[0])
            if isinstance(owner, Glm52ExactTP1BlockFP8FullParamLinear):
                group = "dense_mlp_masters"
            elif isinstance(owner, Glm52FullParamBlockFP8RoutedExperts):
                group = "routed_expert_masters"
            elif isinstance(owner, Glm52ExactFullParamRouterWeight):
                group = "router_masters"
            else:
                raise RuntimeError(
                    f"Parameter {name} looks like a full-param master but its owner "
                    f"{type(owner).__name__} is not an admitted component"
                )
            trained_groups[group] += 1
            trained_bytes[group] += parameter.numel() * parameter.element_size()
            parameter.requires_grad_(True)
            trained += 1
            continue
        parameter.requires_grad_(False)
        frozen += 1
        group = _frozen_group_for(name)
        frozen_groups[group] = frozen_groups.get(group, 0) + 1

    # Fail closed on anything uncovered that is still trainable.
    residual = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.endswith(_TRAINABLE_MASTER_SUFFIXES)
    ]
    if residual:
        raise RuntimeError(f"GLM-5.2 full-param admission left non-allowlisted trainable parameters: {residual[:8]}")
    if trained == 0:
        raise RuntimeError("GLM-5.2 full-param admission installed no trainable masters")

    # The trainable set backpropagates THROUGH the frozen trunk (attention
    # projections, shared experts, out-of-scope banks): admit the validated
    # activation-only backward on every frozen exact-forward
    # module.  Forward bytes are untouched; without this the first frozen
    # boundary above a trainable master raises the scoring-only refusal.
    frozen_dgrad_linears = frozen_dgrad_banks = 0
    for module in model.modules():
        if isinstance(module, NativeBlockFP8Linear):
            module.enable_frozen_activation_dgrad()
            frozen_dgrad_linears += 1
        elif isinstance(module, Glm52NativeBlockFP8Experts) and not isinstance(
            module, Glm52FullParamBlockFP8RoutedExperts
        ):
            module.enable_frozen_activation_dgrad()
            frozen_dgrad_banks += 1
    if frozen_dgrad_linears == 0:
        raise RuntimeError(
            "GLM-5.2 full-param admission found no frozen native block-FP8 projections to admit "
            "for the frozen-trunk activation backward; the model construction is not the admitted row"
        )

    report = Glm52FullParamAdmissionReport(
        contract_version=GLM52_FULLPARAM_ADMISSION_CONTRACT_VERSION,
        dense_mlp_layers=tuple(dense_layers),
        routed_expert_layers=tuple(expert_layers),
        router_layers=tuple(router_layers),
        trained_parameter_count=trained,
        frozen_parameter_count=frozen,
        trained_group_sizes=dict(trained_groups),
        frozen_group_sizes=dict(frozen_groups),
        trained_master_bytes_by_group=dict(trained_bytes),
        adamw_step_bytes_by_group={name: 4 * value for name, value in trained_bytes.items()},
        frozen_dgrad_linear_count=frozen_dgrad_linears,
        frozen_dgrad_bank_count=frozen_dgrad_banks,
    )
    _log_admission_once(report)
    return report


def _validate_expert_scope(
    trainable_expert_layers: Sequence[int] | None,
    plan: Glm52LayerPlan,
) -> frozenset[int] | None:
    """Fail-closed validation of the expert-bank training scope."""

    if trainable_expert_layers is None:
        return None
    if isinstance(trainable_expert_layers, (str, bytes)):
        raise TypeError("trainable_expert_layers must be a sequence of layer indices, not a string")
    sparse_layers = {spec.layer_index for spec in plan.layers if spec.mlp_type is not MLPType.DENSE}
    scope: list[int] = []
    for entry in trainable_expert_layers:
        index = int(entry)
        if index != entry:
            raise TypeError(f"trainable_expert_layers entries must be integers, got {entry!r}")
        if index in scope:
            raise ValueError(f"trainable_expert_layers contains layer {index} more than once")
        if index not in sparse_layers:
            raise ValueError(
                f"trainable_expert_layers names layer {index}, which is not a sparse (MoE) layer of "
                f"this model; sparse layers are {sorted(sparse_layers)}"
            )
        scope.append(index)
    return frozenset(scope)


def _ep_expert_start(local_experts: int, num_global_experts: int) -> int:
    """Global expert offset of this rank's contiguous EP block.

    Mirrors the production ownership rule (``expert_start = ep_rank *
    local_experts``, the canonical-MoE rebasing and the checkpoint pair
    buffer).  A full bank (local == global) is rank 0 by construction; a
    partial bank REQUIRES the model's EP process group — never guess.
    """

    if local_experts == num_global_experts:
        return 0
    import torch.distributed as dist  # noqa: PLC0415

    from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

    ep_group = None
    if dist.is_available() and dist.is_initialized():
        try:
            ep_group = get_parallel_state().ep_group
        except (ValueError, TypeError, KeyError):
            # requires_mesh (no device mesh), no EP mesh matrix, or a mesh
            # without an "ep" dimension: no usable EP group either way.
            ep_group = None
    if ep_group is None:
        raise RuntimeError(
            f"GLM-5.2 full-param admission found an EP-local expert bank ({local_experts} of "
            f"{num_global_experts} experts) but no initialized EP process group; global expert "
            "ownership cannot be derived — refusing to guess"
        )
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    if local_experts * ep_size != num_global_experts:
        raise RuntimeError(
            f"GLM-5.2 full-param admission requires equal contiguous expert slices: "
            f"{local_experts} local x ep_size {ep_size} != {num_global_experts} global"
        )
    return ep_rank * local_experts


def _frozen_group_for(name: str) -> str:
    if ".indexer" in name or ".indexers_proj" in name:
        return "dsa_indexer"
    if ".self_attn." in name:
        return "attention_mla_projections"
    if ".shared_experts." in name:
        return "shared_experts"
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "embeddings"
    if "packed_weight_f32" in name or "weight_scale_inv" in name or "quantized_weight" in name:
        return "fp8_byte_caches (never trainable)"
    if ".gate." in name or name.endswith("gate.weight"):
        return "legacy router tensors"
    return "norms_biases_and_other"


def assert_glm52_fullparam_allowlist(model: nn.Module) -> None:
    """Re-verify the trainable set (call after any external mutation)."""

    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.endswith(_TRAINABLE_MASTER_SUFFIXES)
    ]
    if violations:
        raise RuntimeError(
            f"GLM-5.2 full-param allowlist violated; non-admitted trainable parameters: {violations[:8]}"
        )


@torch.no_grad()
def refresh_glm52_fullparam_caches(model: nn.Module) -> int:
    """Collective-safe, bounded-memory step-boundary cache refresh.

    All rank-local preconditions and the ordered component inventory are
    rendezvoused before the first DTensor all-gather. Every master is then
    gathered, staged, and committed one component at a time so a full model's
    unsharded FP32 masters are never retained simultaneously. If a rank-local
    staging/commit failure occurs, that rank records it but still participates
    in every later master gather; a final error rendezvous then makes all ranks
    fail the optimizer command and restart rather than use partial caches.
    """

    plan: list[tuple[str, nn.Module, str]] = []
    local_error: Exception | None = None
    try:
        for name, module in model.named_modules():
            if isinstance(module, Glm52FullParamDenseMLP):
                kind = "dense_mlp"
            elif isinstance(module, Glm52FullParamBlockFP8RoutedExperts):
                kind = "routed_experts"
            elif isinstance(module, Glm52ExactFullParamRouterWeight):
                kind = "router"
            else:
                continue
            module._validate_refresh_source()
            plan.append((name, module, kind))
        if not plan:
            raise RuntimeError("refresh_glm52_fullparam_caches found no admitted full-param components")
    except Exception as exc:
        local_error = exc

    signature = tuple((name, kind, type(module).__name__) for name, module, kind in plan)
    if dist.is_available() and dist.is_initialized():
        local_status = {
            "error": None if local_error is None else f"{type(local_error).__name__}: {str(local_error)[:512]}",
            "signature": signature,
        }
        gathered: list[dict] = [{} for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_status)
        failures = [
            f"rank {rank}: {status.get('error')}"
            for rank, status in enumerate(gathered)
            if status.get("error") is not None
        ]
        if failures:
            raise RuntimeError("GLM-5.2 full-param refresh preflight failed collectively: " + "; ".join(failures))
        signatures = {repr(status.get("signature")) for status in gathered}
        if len(signatures) != 1:
            raise RuntimeError(
                "GLM-5.2 full-param refresh component order differs across ranks; refusing DTensor gathers"
            )
    elif local_error is not None:
        raise local_error

    refresh_error: Exception | None = None
    refreshed = 0
    for name, module, kind in plan:
        masters = None
        staged = None
        try:
            if kind in {"dense_mlp", "routed_experts"}:
                masters = module._gather_refresh_masters()
            else:
                masters = module._gather_refresh_master()
        except Exception as exc:
            # A DTensor collective failure is already a broken distributed
            # session. Preserve the first cause; peers will either observe the
            # same collective error or terminate via the runner's fatal path.
            if refresh_error is None:
                refresh_error = RuntimeError(f"GLM-5.2 full-param refresh gather failed for {name!r} ({kind})")
                refresh_error.__cause__ = exc

        if refresh_error is None:
            try:
                if kind in {"dense_mlp", "routed_experts"}:
                    staged = module._stage_quantized_caches(masters)
                    module._commit_quantized_caches(staged)
                else:
                    staged = module._stage_effective_view(masters)
                    module._commit_effective_view(staged)
                refreshed += 1
            except Exception as exc:
                refresh_error = RuntimeError(f"GLM-5.2 full-param refresh staging/commit failed for {name!r} ({kind})")
                refresh_error.__cause__ = exc
        # Drop the only full-master and replacement-cache references before
        # the next component's all-gather. The CUDA caching allocator may
        # retain storage, but it can reuse it instead of growing model-wide.
        del masters
        del staged

    if dist.is_available() and dist.is_initialized():
        local_message = None if refresh_error is None else f"{type(refresh_error).__name__}: {str(refresh_error)[:512]}"
        gathered_errors: list[str | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_errors, local_message)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(gathered_errors) if message is not None]
        if failures:
            raise RuntimeError(
                "GLM-5.2 full-param refresh failed collectively after completing the gather schedule: "
                + "; ".join(failures)
            ) from refresh_error
    elif refresh_error is not None:
        raise refresh_error
    return refreshed


def rebind_glm52_fullparam_master_identities(model: nn.Module) -> int:
    """Re-stamp every admitted master identity after an identity-changing,
    VALUE-preserving structural transform (the FSDP2 wrap).

    The admission seeds masters and caches from checkpoint bytes BEFORE
    ``fully_shard`` wraps the model; wrapping replaces the master parameters'
    storage (plain tensor -> DTensor), so the pre-wrap freshness bindings can
    never match again and the first forward would fail the staleness gate.
    Cache bytes are deliberately NOT touched: initial caches must remain the
    checkpoint bytes bitwise; a refresh here would replace them with
    Q(master), whose scales differ from the checkpoint's. Only the freshness
    binding moves to the post-transform storage.

    Call this exactly once, immediately after the wrap and before any
    forward or optimizer step; anywhere else it would mask real staleness.
    """

    rebound = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                Glm52ExactTP1BlockFP8FullParamLinear,
                Glm52FullParamBlockFP8RoutedExperts,
                Glm52ExactFullParamRouterWeight,
            ),
        ):
            module._record_master_identity()
            rebound += 1
    if rebound == 0:
        raise RuntimeError("rebind_glm52_fullparam_master_identities found no admitted full-param components")
    return rebound


def restore_glm52_fullparam_master_identities(model: nn.Module) -> int:
    """Re-stamp freshness after a successful strict full-model DCP restore.

    A full restore writes both FP32 masters and their serving-byte caches from
    the same checkpoint.  Those writes legitimately increment the masters'
    tensor versions, so the pre-load freshness bindings become stale even
    though the restored pair is coherent.  Record the restored identities
    without refreshing or requantizing: checkpoint cache bytes are the source
    of truth and must remain bitwise unchanged.

    This helper belongs only after the strict load transaction has returned
    successfully.  Calling it after a partial or failed load could bless a
    mixed state and is therefore intentionally left to the ModelRunner's
    post-success hook.
    """

    restored = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                Glm52ExactTP1BlockFP8FullParamLinear,
                Glm52FullParamBlockFP8RoutedExperts,
                Glm52ExactFullParamRouterWeight,
            ),
        ):
            module._record_master_identity()
            restored += 1
    if restored == 0:
        raise RuntimeError("restore_glm52_fullparam_master_identities found no admitted full-param components")
    return restored


def iter_glm52_fullparam_publication(model: nn.Module) -> list[tuple[str, object]]:
    """Enumerate (target, publication unit) pairs in checkpoint form.

    Targets are checkpoint FQNs: dense composites publish ONE
    ``block_fp8_dense_mlp`` item (the split gate/up/down tensors the serving
    loader consumes); expert banks publish one globally-indexed
    ``block_fp8_expert`` item per local expert
    (``{bank_fqn}.{global_expert_id}``); routers publish their BF16 view.
    Full-param linears inside a composite are covered by the composite and
    never enumerated separately; a standalone linear (component rigs only)
    still publishes its fused byte pair.  Deterministic name order.
    """

    components: list[tuple[str, object]] = []
    composite_prefixes: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, Glm52FullParamDenseMLP):
            components.append((name, module))
            composite_prefixes.append(f"{name}." if name else "")
        elif isinstance(module, Glm52FullParamBlockFP8RoutedExperts):
            components.extend(
                (f"{name}.{global_id}", publication) for global_id, publication in module.checkpoint_publications()
            )
        elif isinstance(module, Glm52ExactFullParamRouterWeight):
            components.append((name, module))
    for name, module in model.named_modules():
        if isinstance(module, Glm52ExactTP1BlockFP8FullParamLinear) and not any(
            name.startswith(prefix) for prefix in composite_prefixes
        ):
            components.append((name, module))
    if not components:
        raise RuntimeError("iter_glm52_fullparam_publication found no admitted full-param components")
    return sorted(components, key=lambda entry: entry[0])


def _validate_official_fullparam_config(config) -> None:
    expected = {
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    mismatches = [
        f"{name}={getattr(config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    if tuple(getattr(config, "mlp_layer_types", ()) or ()) != ("dense",) * 3 + ("sparse",) * 75:
        mismatches.append("mlp_layer_types does not match the official 3 dense + 75 sparse schedule")
    if mismatches:
        raise ValueError(
            "GLM-5.2 full-param training supports only the official model geometry: " + ", ".join(mismatches)
        )
    if getattr(config, "_ep_dispatch", None) != "alltoall":
        raise ValueError("GLM-5.2 full-param training requires ep_dispatch='alltoall' (DeepEP is rejected)")
    if getattr(config, "_glm52_exact_contract", False):
        raise ValueError(
            "GLM-5.2 full-param training is a training lane and cannot use the scoring-only exact contract flag"
        )


def prepare_glm52_fullparam_training(
    model: nn.Module,
    config,
    *,
    trainable_expert_layers: Sequence[int],
) -> Glm52FullParamAdmissionReport:
    """Production entry point: official geometry only, fail closed everywhere.

    ``trainable_expert_layers`` is required and must be non-empty so production
    states its memory-bounded expert scope explicitly instead of inheriting an
    implicit "all".
    """

    if trainable_expert_layers is None or isinstance(trainable_expert_layers, (str, bytes)):
        raise TypeError("prepare_glm52_fullparam_training requires an explicit trainable_expert_layers sequence")
    if len(trainable_expert_layers) == 0:
        raise ValueError(
            "prepare_glm52_fullparam_training requires a non-empty trainable_expert_layers scope; "
            "a dense/router-only configuration is not an admitted production mode"
        )
    return install_glm52_fullparam_components(model, config, trainable_expert_layers=trainable_expert_layers)


def _log_admission_once(report: Glm52FullParamAdmissionReport) -> None:
    global _engagement_logged
    if _engagement_logged:
        return
    _engagement_logged = True
    trained_summary = ", ".join(f"{name}={count}" for name, count in sorted(report.trained_group_sizes.items()))
    frozen_summary = ", ".join(f"{name!r}={count}" for name, count in sorted(report.frozen_group_sizes.items()))
    envelope_summary = ", ".join(
        f"{name}={report.trained_master_bytes_by_group[name] / 2**30:.3f}GiB"
        f"(adamw {report.adamw_step_bytes_by_group[name] / 2**30:.3f}GiB)"
        for name in sorted(report.trained_master_bytes_by_group)
    )
    logger.info(
        "GLM-5.2 full-param admission engaged: contract=%s dense_layers=%s expert_layers=%s router_layers=%s | "
        "TRAINED (%d params): %s | FROZEN (%d params): %s | rank memory envelope: %s | "
        "frozen-trunk dgrad admitted: linears=%d banks=%d",
        report.contract_version,
        list(report.dense_mlp_layers),
        list(report.routed_expert_layers),
        list(report.router_layers),
        report.trained_parameter_count,
        trained_summary,
        report.frozen_parameter_count,
        frozen_summary,
        envelope_summary,
        report.frozen_dgrad_linear_count,
        report.frozen_dgrad_bank_count,
    )


__all__ = [
    "GLM52_FULLPARAM_ADMISSION_CONTRACT_VERSION",
    "Glm52FullParamAdmissionReport",
    "Glm52FullParamTopkRouter",
    "assert_glm52_fullparam_allowlist",
    "install_glm52_fullparam_components",
    "iter_glm52_fullparam_publication",
    "prepare_glm52_fullparam_training",
    "rebind_glm52_fullparam_master_identities",
    "refresh_glm52_fullparam_caches",
    "restore_glm52_fullparam_master_identities",
]
