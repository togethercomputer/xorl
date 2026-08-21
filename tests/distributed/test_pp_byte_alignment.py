"""PP1-vs-PP2 byte-equality gate for the trainer's logit path.

The decisive byte-boundary check runs the same tokens through the
unpartitioned (PP1) program and through a real PP2
split built with the production machinery (``pipeline_module_split`` +
``PipelineStage`` + ``forward_only_pp``), and the logit-path outputs must be
BYTE identical (integer-view ``torch.equal``), not merely close.

Modes (one torchrun launch each):

- ``generic``: native RMSNorm + eager attention + eager CE. Pins the PP
  plumbing itself (send/recv, pruning, metadata routing, stage-local RoPE).
- ``exact``: ``_qwen35_exact_contract`` v1 program — ``sglang_fused`` BI
  RMSNorm kernels, FA4 varlen attention, exact-contract GDN, ``batch_invariant``
  head with ``lm_head_fp32``. Pins the contract lane, and additionally gates
  microbatch composition invariance, PP-mandated padding, and fail-closed
  metadata handling.
- ``exact_fsdp``: exact program with FSDP2-wrapped stage parts (the
  production shape) — wrapping must not change forward bytes.
- ``exact_interleaved``: exact program under Interleaved1F1B with 2 virtual
  stages per rank (4 cuts) — schedule and extra cuts must not change bytes.

Requires 2 GPUs. Byte equality is asserted on the last-stage rank and the
verdict table is gathered to every rank before any assertion so a failure
cannot deadlock the peer.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    build_pp_stage,
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)
from xorl.distributed.pp_byte_contract import PPByteContractError
from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from xorl.objectives.causallm_loss import causallm_loss_function
from xorl.trainers.training_utils import forward_only_pp, pad_micro_batches_for_pp
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

SEQ_LEN = 32
PAD_SEQ_LEN = 48
N_MICROBATCHES = 2
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
# 4 virtual stages need >= 1 decoder layer per stage after the weighted FQN
# split (the byte contract refuses layer-less stages), so run 8 layers there.
LAYER_TYPES_INTERLEAVED = LAYER_TYPES * 2
IGNORE_INDEX = -100

# Required checks per mode; every listed check must be True in the gathered
# verdict table, and no unlisted check may fail either (the gate fails closed
# on anything it measured).
_REQUIRED = {
    "generic": ("core_hidden", "core_logprobs"),
    "exact": (
        "core_hidden",
        "core_logprobs",
        "contract_marked",
        "composition_merged_hidden",
        "composition_merged_logprobs",
        "padding_hidden",
        "padding_logprobs",
        "metadata_raises",
    ),
    # Production shape: FSDP2-wrapped stage parts (bf16 params, per-layer +
    # root units) must not change forward bytes vs the unwrapped PP1 program.
    "exact_fsdp": ("core_hidden", "core_logprobs", "contract_marked"),
    # Loop-style virtual stages: 4 stages on 2 ranks under Interleaved1F1B —
    # more cuts, a different schedule, same bytes required.
    "exact_interleaved": ("core_hidden", "core_logprobs", "contract_marked"),
}

_EXACT_MODES = frozenset({"exact", "exact_fsdp", "exact_interleaved"})


def _stage_layout(mode: str, pp_size: int) -> tuple:
    """(num_stages, stage_style, schedule_name) for the mode."""
    if mode == "exact_interleaved":
        return 2 * pp_size, "loop", "Interleaved1F1B"
    return pp_size, "single", "1F1B"


def _build_config(mode: str) -> Qwen3_5Config:
    layer_types = LAYER_TYPES_INTERLEAVED if mode == "exact_interleaved" else LAYER_TYPES
    config = Qwen3_5Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=list(layer_types),
        max_position_embeddings=256,
        use_cache=False,
        tie_word_embeddings=False,
    )
    if mode in _EXACT_MODES:
        # The exact v1 value program: BI RMSNorm families + FA4 + batch_invariant head.
        set_rmsnorm_mode("sglang_fused")
        config._attn_implementation = "flash_attention_4"
        config._qwen35_exact_contract = True
        config._qwen35_rmsnorm_family = "v1"
        # The byte contract fails closed on undeclared dtype.
        config.dtype = torch.bfloat16
    else:
        set_rmsnorm_mode("native")
        config._attn_implementation = "eager"
    return config


def _ce_kwargs(mode: str) -> dict:
    if mode in _EXACT_MODES:
        return {"ce_mode": "batch_invariant", "lm_head_fp32": True}
    return {"ce_mode": "eager"}


def _make_microbatch(seed: int, seq_len: int, vocab_size: int, device: torch.device) -> dict:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=generator).to(device)
    labels = torch.roll(input_ids, -1, dims=-1).clone()
    labels[:, -1] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        "cu_seq_lens_k": torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        "max_length_q": seq_len,
        "max_length_k": seq_len,
    }


_INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int64: torch.int64}


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view_dtype = _INT_VIEW[a.dtype]
    return bool(torch.equal(a.detach().contiguous().view(view_dtype), b.detach().contiguous().view(view_dtype)))


def _pp1_hidden(model, mb: dict) -> torch.Tensor:
    """The unpartitioned program's logit-path hidden for one microbatch."""
    with torch.no_grad():
        outputs = model(
            input_ids=mb["input_ids"],
            position_ids=mb["position_ids"],
            use_cache=False,
            output_hidden_states=False,
            cu_seq_lens_q=mb["cu_seq_lens_q"],
            cu_seq_lens_k=mb["cu_seq_lens_k"],
            max_length_q=mb["max_length_q"],
            max_length_k=mb["max_length_k"],
        )
    return outputs.last_hidden_state


def _head_logprobs(hidden: torch.Tensor, lm_head_weight: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    if hasattr(lm_head_weight, "full_tensor"):
        # FSDP2-wrapped parts reshard after forward under no_grad; materialize
        # the full bf16 weight for the head (mirrors the production PP loop).
        lm_head_weight = lm_head_weight.full_tensor()
    with torch.no_grad():
        result = causallm_loss_function(
            hidden_states=hidden,
            weight=lm_head_weight,
            labels=labels,
            return_per_token=True,
            **_ce_kwargs(mode),
        )
    return result.per_token_logprobs


def _fsdp_wrap_parts(model_parts, ps) -> None:
    """FSDP2-wrap each stage part: per-decoder-layer units plus a root unit,
    no mixed-precision policy, so compute dtypes equal the unwrapped
    reference and the mode isolates FSDP sharding mechanics against byte
    identity. The exact GDN pins A_log/dt_bias to fp32 (its ``.to(bf16)`` is
    a deliberate no-op there); FSDP2 requires uniform original dtypes per
    unit, so those pins ride along as ignored_params. The full production
    shape (fp32 masters + bf16 MP policy) is covered at real geometry by the
    full-model endpoint replay."""
    from torch.distributed._composable.fsdp import fully_shard  # noqa: PLC0415

    mesh = ps.fsdp_mesh

    def _wrap(module) -> None:
        pinned = {p for p in module.parameters() if p.dtype != torch.bfloat16}
        fully_shard(module, mesh=mesh, ignored_params=pinned)

    for part in model_parts:
        for layer in part.model.layers:
            if layer is not None:
                _wrap(layer)
        _wrap(part)


def _stage_io(stage_index: int, mbs: int, seq_len: int, config) -> tuple:
    """Meta IO so PipelineStage skips shape inference (mirrors _build_pp_stage_io)."""
    if stage_index == 0:
        input_args = (torch.empty(mbs, seq_len, dtype=torch.long, device="meta"),)
    else:
        input_args = (torch.empty(mbs, seq_len, config.hidden_size, dtype=torch.bfloat16, device="meta"),)
    # Forward-only: the last stage returns HIDDEN (lm_head applied outside).
    output_args = (torch.empty(mbs, seq_len, config.hidden_size, dtype=torch.bfloat16, device="meta"),)
    return input_args, output_args


def _build_forward_only_schedule(model_parts, init_stages, seq_len: int, config, ps, device, schedule_name: str):
    num_stages = init_stages[0].num_stages
    stages = []
    for model_part, init_stage in zip(model_parts, init_stages):
        model_part._pp_lm_head_in_loss = True
        input_args, output_args = _stage_io(init_stage.stage_index, 1, seq_len, config)
        stages.append(
            build_pp_stage(
                model_part,
                stage_index=init_stage.stage_index,
                num_stages=num_stages,
                device=device,
                pp_group=ps.pp_group,
                input_args=input_args,
                output_args=output_args,
            )
        )
    return build_pipeline_schedule(stages, n_microbatches=N_MICROBATCHES, loss_fn=None, schedule_name=schedule_name)


def _run_pp2_forward_only(model_parts, init_stages, micro_batches, config, ps, device, schedule_name: str):
    seq_len = micro_batches[0]["input_ids"].shape[-1]
    num_stages = init_stages[0].num_stages
    schedule = _build_forward_only_schedule(model_parts, init_stages, seq_len, config, ps, device, schedule_name)
    with torch.no_grad():
        return forward_only_pp(
            model_parts=model_parts,
            pp_schedule=schedule,
            micro_batches=micro_batches,
            has_first_stage=any(s.stage_index == 0 for s in init_stages),
            has_last_stage=any(s.stage_index == num_stages - 1 for s in init_stages),
        )


def _merged_packed_batch(micro_batches: list, device: torch.device) -> dict:
    """Pack the same documents into one forward for the composition gate."""
    seq_lens = [mb["input_ids"].shape[-1] for mb in micro_batches]
    boundaries = [0]
    for seq_len in seq_lens:
        boundaries.append(boundaries[-1] + seq_len)
    cu = torch.tensor(boundaries, dtype=torch.int32, device=device)
    return {
        "input_ids": torch.cat([mb["input_ids"] for mb in micro_batches], dim=-1),
        "labels": torch.cat([mb["labels"] for mb in micro_batches], dim=-1),
        "position_ids": torch.cat([mb["position_ids"] for mb in micro_batches], dim=-1),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": max(seq_lens),
        "max_length_k": max(seq_lens),
    }


def _clone_microbatches(micro_batches: list) -> list:
    return [{k: v.clone() if torch.is_tensor(v) else v for k, v in mb.items()} for mb in micro_batches]


def _record(verdicts: dict, name: str, ok: bool, detail: str = "") -> None:
    verdicts[name] = (bool(ok), detail)


def _run_gate(mode: str) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=2,
        ulysses_size=1,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    ps = get_parallel_state()
    device = torch.device("cuda", local_rank)
    num_stages, stage_style, schedule_name = _stage_layout(mode, ps.pp_size)

    config = _build_config(mode)
    torch.manual_seed(1234)
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).to(device).eval()
    # Identical weight bytes on every rank regardless of CPU init determinism.
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)

    micro_batches = [_make_microbatch(100 + i, SEQ_LEN, config.vocab_size, device) for i in range(N_MICROBATCHES)]

    # ---- PP2 split via the production machinery -------------------------
    pp_config = model.get_pp_module_config()
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=num_stages,
        num_layers=pp_config["num_layers"],
        input_fqns=pp_config["input_fqns"],
        layer_prefix=pp_config["layer_prefix"],
        output_fqns=pp_config["output_fqns"],
    )
    init_stages, model_parts = pipeline_module_split(
        model,
        pp_mesh=ps.pp_mesh,
        device=device,
        module_names_per_stage=module_names_per_stage,
        always_keep_fqns=pp_config["always_keep_fqns"],
        stage_style=stage_style,
    )
    for model_part in model_parts:
        model_part.eval()
    if mode == "exact_fsdp":
        _fsdp_wrap_parts(model_parts, ps)
    is_last_stage = any(s.stage_index == num_stages - 1 for s in init_stages)

    verdicts: dict = {}
    if mode in _EXACT_MODES:
        marked = all(getattr(part, "_pp_exact_boundary_contract", False) for part in model_parts)
        _record(verdicts, "contract_marked", marked, "engage_pp_byte_contract marked all local parts")

    # ---- PP1 reference (the uncut program, run locally on every rank) ---
    ref_hidden = [_pp1_hidden(model, mb) for mb in micro_batches]

    # ---- Core gate: PP2 schedule vs PP1 bytes ----------------------------
    hidden_per_mb = _run_pp2_forward_only(model_parts, init_stages, micro_batches, config, ps, device, schedule_name)

    if is_last_stage:
        try:
            lm_head_weight = model_parts[-1].lm_head.weight
            hidden_ok = all(_same_bytes(pp2, ref) for pp2, ref in zip(hidden_per_mb, ref_hidden))
            _record(verdicts, "core_hidden", hidden_ok, "PP2 last-stage hidden bytes == PP1 hidden bytes")
            logprob_ok = True
            for pp2_hidden, ref_h, mb in zip(hidden_per_mb, ref_hidden, micro_batches):
                pp2_lp = _head_logprobs(pp2_hidden, lm_head_weight, mb["labels"], mode)
                ref_lp = _head_logprobs(ref_h, model.lm_head.weight, mb["labels"], mode)
                logprob_ok = logprob_ok and _same_bytes(pp2_lp, ref_lp)
            _record(verdicts, "core_logprobs", logprob_ok, "PP2 per-token FP32 logprob bytes == PP1")
        except Exception as exc:  # record, never deadlock the peer
            _record(verdicts, "core_hidden", False, f"exception: {type(exc).__name__}: {exc}")
            _record(verdicts, "core_logprobs", False, "not evaluated after exception")

    if mode == "exact":
        # ---- Microbatch composition invariance (PP1, packed merge) -----
        if is_last_stage:
            try:
                merged = _merged_packed_batch(micro_batches, device)
                merged_hidden = _pp1_hidden(model, merged)
                segments_ok = True
                offset = 0
                for ref_h, mb in zip(ref_hidden, micro_batches):
                    seq_len = mb["input_ids"].shape[-1]
                    segment = merged_hidden[:, offset : offset + seq_len]
                    segments_ok = segments_ok and _same_bytes(segment, ref_h)
                    offset += seq_len
                _record(
                    verdicts,
                    "composition_merged_hidden",
                    segments_ok,
                    "merged packed batch hidden bytes == per-microbatch",
                )
                merged_lp = _head_logprobs(merged_hidden, model.lm_head.weight, merged["labels"], mode)
                lp_ok = True
                offset = 0
                for ref_h, mb in zip(ref_hidden, micro_batches):
                    seq_len = mb["input_ids"].shape[-1]
                    ref_lp = _head_logprobs(ref_h, model.lm_head.weight, mb["labels"], mode)
                    lp_ok = lp_ok and _same_bytes(merged_lp[:, offset : offset + seq_len], ref_lp)
                    offset += seq_len
                _record(
                    verdicts,
                    "composition_merged_logprobs",
                    lp_ok,
                    "merged packed batch logprob bytes == per-microbatch",
                )
            except Exception as exc:
                _record(
                    verdicts,
                    "composition_merged_hidden",
                    False,
                    f"exception: {type(exc).__name__}: {exc}",
                )
                _record(verdicts, "composition_merged_logprobs", False, "not evaluated after exception")

        # ---- PP-mandated padding (padded PP2 vs unpadded PP1) ----------
        padded = _clone_microbatches(micro_batches)
        pad_micro_batches_for_pp(padded, sample_packing_sequence_len=PAD_SEQ_LEN, sp_size=1)
        padded_hidden_per_mb = _run_pp2_forward_only(
            model_parts, init_stages, padded, config, ps, device, schedule_name
        )
        if is_last_stage:
            try:
                lm_head_weight = model_parts[-1].lm_head.weight
                pad_hidden_ok = True
                pad_lp_ok = True
                for pad_hidden, ref_h, pad_mb, mb in zip(padded_hidden_per_mb, ref_hidden, padded, micro_batches):
                    real = mb["input_ids"].shape[-1]
                    pad_hidden_ok = pad_hidden_ok and _same_bytes(pad_hidden[:, :real], ref_h)
                    pad_lp = _head_logprobs(pad_hidden, lm_head_weight, pad_mb["labels"], mode)
                    ref_lp = _head_logprobs(ref_h, model.lm_head.weight, mb["labels"], mode)
                    valid = (mb["labels"] != IGNORE_INDEX).view(-1)
                    pad_lp_ok = pad_lp_ok and _same_bytes(
                        pad_lp.view(-1)[: valid.shape[0]][valid], ref_lp.view(-1)[valid]
                    )
                _record(
                    verdicts,
                    "padding_hidden",
                    pad_hidden_ok,
                    "padded PP2 real-token hidden bytes == unpadded PP1",
                )
                _record(
                    verdicts,
                    "padding_logprobs",
                    pad_lp_ok,
                    "padded PP2 real-token logprob bytes == unpadded PP1",
                )
            except Exception as exc:
                _record(verdicts, "padding_hidden", False, f"exception: {type(exc).__name__}: {exc}")
                _record(verdicts, "padding_logprobs", False, "not evaluated after exception")

        # ---- Metadata starvation must raise on marked parts (local) ----
        try:
            part = model_parts[0]
            part._pp_batch_metadata = deque()
            probe = (
                micro_batches[0]["input_ids"]
                if part._pp_is_first
                else torch.zeros(1, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16, device=device)
            )
            part._pp_forward_only = True
            try:
                part.forward(probe)
                _record(verdicts, "metadata_raises", False, "scheduled forward without metadata did NOT raise")
            except PPByteContractError as exc:
                _record(verdicts, "metadata_raises", True, f"raised as required: {exc}")
            finally:
                part._pp_forward_only = False
        except Exception as exc:
            _record(verdicts, "metadata_raises", False, f"unexpected: {type(exc).__name__}: {exc}")

    # ---- Gather verdicts everywhere BEFORE asserting anything -----------
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, verdicts)
    merged_verdicts: dict = {}
    for rank_verdicts in gathered:
        merged_verdicts.update(rank_verdicts or {})

    if dist.get_rank() == 0:
        print(f"=== PP byte-alignment gate verdicts (mode={mode}) ===", flush=True)
        for name in sorted(merged_verdicts):
            ok, detail = merged_verdicts[name]
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    missing = [name for name in _REQUIRED[mode] if name not in merged_verdicts]
    failed = [name for name, (ok, _) in merged_verdicts.items() if not ok]
    if missing or failed:
        raise AssertionError(
            f"PP byte-alignment gate (mode={mode}) failed: missing={missing} failed="
            f"{[(name, merged_verdicts[name][1]) for name in failed]}"
        )
    if dist.get_rank() == 0:
        print(f"PP byte-alignment gate passed (mode={mode})", flush=True)


def _main() -> None:
    mode = os.environ.get("PP_GATE_MODE", "generic")
    if mode not in _REQUIRED:
        raise ValueError(f"Unsupported PP_GATE_MODE: {mode}")
    _run_gate(mode)
    # No destroy_process_group here: with FSDP2-wrapped stage parts the NCCL
    # teardown wedges on the lazily created P2P/subgroup communicators (both
    # ranks spin in destroy after every check passed). The harness exits and
    # torchrun reaps; the verdict is the exit code plus the flushed table.


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_pp2_byte_alignment_generic_program():
        result = run_distributed_script(__file__, num_gpus=2, timeout=600, extra_env={"PP_GATE_MODE": "generic"})
        result.assert_success("PP2 must be byte-identical to PP1 for the generic plumbing")

    @skip_if_gpu_count_less_than(2)
    def test_pp2_byte_alignment_exact_program():
        result = run_distributed_script(__file__, num_gpus=2, timeout=600, extra_env={"PP_GATE_MODE": "exact"})
        result.assert_success("PP2 must be byte-identical to PP1 for the exact value program")

    @skip_if_gpu_count_less_than(2)
    def test_pp2_byte_alignment_exact_fsdp_wrapped_stages():
        result = run_distributed_script(__file__, num_gpus=2, timeout=600, extra_env={"PP_GATE_MODE": "exact_fsdp"})
        result.assert_success("FSDP2-wrapped PP2 stages must be byte-identical to unwrapped PP1")

    @skip_if_gpu_count_less_than(2)
    def test_pp2_byte_alignment_exact_interleaved_schedule():
        result = run_distributed_script(
            __file__, num_gpus=2, timeout=600, extra_env={"PP_GATE_MODE": "exact_interleaved"}
        )
        result.assert_success("Interleaved1F1B virtual stages must be byte-identical to PP1")


if __name__ == "__main__":
    _main()
