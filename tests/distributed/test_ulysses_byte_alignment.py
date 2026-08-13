"""Ulysses U1-vs-UN byte-equality gate for the exact logit path.

Two-phase structure (the Ulysses strategy reads GLOBAL parallel state, so the
U1 reference must run in its own process):

- phase "ref" (1 GPU, no torch.distributed): the exact-contract program on a
  FULL-ATTENTION-ONLY tiny model with the production head geometry
  (8 Q-heads / 2 KV-heads / head_dim 256, bf16, FA4, sglang_fused BI RMSNorm,
  bi_fused head) over packed varlen inputs; writes last-hidden and per-token
  logprob bytes to an npz.
- phase "shard" (torchrun, ULYSSES_GATE_DEGREE ranks): the same model and
  tokens through the production Ulysses path (sequence-sharded input_ids,
  FULL position_ids and cu_seqlens per the collator convention); gathers the
  sequence shards in rank order and byte-compares hidden + logprobs against
  the reference npz. Includes the collator cp-multiple padding case
  (pad tokens appended as their own documents; real-token bytes must match
  the unpadded U1 reference) and the hybrid negative: a GDN layer under
  Ulysses must RAISE the exact-contract CP refusal, not compute.

"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from xorl.ops.loss.causallm_loss import causallm_loss_function


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

SEQ_LEN = 512  # divisible by every tested degree
# Padding case: 505 is odd, so EVERY tested degree needs collator padding
# (degree 2 -> pad 1, degree 8 -> pad 7).
SHORT_LEN = SEQ_LEN - 7
DOC_BOUNDARY = 192  # two packed documents
IGNORE_INDEX = -100


def _build_config(layer_types) -> Qwen3_5Config:
    set_rmsnorm_mode("sglang_fused")
    config = Qwen3_5Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(layer_types),
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=list(layer_types),
        max_position_embeddings=2048,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_4"
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v1"
    config.dtype = torch.bfloat16
    return config


def _build_model(layer_types, device):
    torch.manual_seed(1729)
    config = _build_config(layer_types)
    return Qwen3_5ForCausalLM(config).to(torch.bfloat16).to(device).eval(), config


def _make_batch(seq_len: int, vocab_size: int, device):
    generator = torch.Generator().manual_seed(777)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=generator).to(device)
    labels = torch.roll(input_ids, -1, dims=-1).clone()
    labels[:, -1] = IGNORE_INDEX
    cu = torch.tensor([0, DOC_BOUNDARY, seq_len], dtype=torch.int32, device=device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": seq_len - DOC_BOUNDARY,
        "max_length_k": seq_len - DOC_BOUNDARY,
    }


def _pad_like_collator(batch: dict, cp_size: int, device):
    """Mirror sequence_shard_collator: pad the packed length to a multiple of
    cp_size; pad tokens get sequential position ids and their own document."""
    seq_len = batch["input_ids"].shape[-1]
    target = (seq_len + cp_size - 1) // cp_size * cp_size
    pad = target - seq_len
    if pad == 0:
        return batch
    out = dict(batch)
    out["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, pad), value=0)
    out["labels"] = torch.nn.functional.pad(batch["labels"], (0, pad), value=IGNORE_INDEX)
    out["position_ids"] = torch.cat(
        [batch["position_ids"], torch.arange(pad, device=device).unsqueeze(0)], dim=-1
    )
    cu = batch["cu_seq_lens_q"].tolist() + [target]
    out["cu_seq_lens_q"] = torch.tensor(cu, dtype=torch.int32, device=device)
    out["cu_seq_lens_k"] = out["cu_seq_lens_q"].clone()
    return out


def _forward_hidden(model, batch):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            use_cache=False,
            output_hidden_states=False,
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_k"],
            max_length_q=batch["max_length_q"],
            max_length_k=batch["max_length_k"],
        )
    return outputs.last_hidden_state


def _logprobs(model, hidden, labels):
    with torch.no_grad():
        result = causallm_loss_function(
            hidden_states=hidden,
            weight=model.lm_head.weight,
            labels=labels,
            return_per_token=True,
            ce_mode="bi_fused",
            lm_head_fp32=True,
        )
    return result.per_token_logprobs


def _short_batch(batch: dict, device):
    short = dict(batch)
    for key in ("input_ids", "labels", "position_ids"):
        short[key] = batch[key][:, :SHORT_LEN].contiguous()
    short["cu_seq_lens_q"] = torch.tensor([0, DOC_BOUNDARY, SHORT_LEN], dtype=torch.int32, device=device)
    short["cu_seq_lens_k"] = short["cu_seq_lens_q"].clone()
    short["max_length_q"] = SHORT_LEN - DOC_BOUNDARY
    short["max_length_k"] = SHORT_LEN - DOC_BOUNDARY
    return short


def _run_reference(out_path: str) -> None:
    device = torch.device("cuda")
    model, config = _build_model(["full_attention"] * 4, device)
    batch = _make_batch(SEQ_LEN, config.vocab_size, device)
    hidden = _forward_hidden(model, batch)
    logprobs = _logprobs(model, hidden, batch["labels"])
    # The padding case compares against a REAL unpadded short-sequence run,
    # not a truncated slice of the long run (truncation-bitwise-equivalence
    # would itself be an unproven assumption).
    short = _short_batch(batch, device)
    hidden_short = _forward_hidden(model, short)
    np.savez(
        out_path,
        hidden_bf16=hidden.view(torch.int16).cpu().numpy(),
        logprobs=logprobs.view(torch.int32).cpu().numpy(),
        hidden_short_bf16=hidden_short.view(torch.int16).cpu().numpy(),
    )
    print(f"[ulysses-gate] reference written: {out_path}", flush=True)


def _run_sharded() -> None:
    import torch.distributed as dist

    from xorl.distributed.parallel_state import init_parallel_state
    from xorl.utils.device import get_nccl_backend

    degree = int(os.environ["ULYSSES_GATE_DEGREE"])
    ref_path = os.environ["ULYSSES_GATE_REF"]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=degree,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()

    model, config = _build_model(["full_attention"] * 4, device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)

    reference = np.load(ref_path)
    verdicts = {}

    def _sharded_forward(batch):
        seq_len = batch["input_ids"].shape[-1]
        assert seq_len % degree == 0
        shard = seq_len // degree
        local_ids = batch["input_ids"][:, rank * shard : (rank + 1) * shard].contiguous()
        local_batch = dict(batch)
        local_batch["input_ids"] = local_ids
        hidden_local = _forward_hidden(model, local_batch)
        gathered = [torch.empty_like(hidden_local) for _ in range(degree)]
        dist.all_gather(gathered, hidden_local.contiguous())
        return torch.cat(gathered, dim=1)

    # --- core: clean divisible packed batch --------------------------------
    batch = _make_batch(SEQ_LEN, config.vocab_size, device)
    hidden_full = _sharded_forward(batch)
    hidden_ok = bool(
        torch.equal(
            hidden_full.view(torch.int16).cpu(), torch.from_numpy(reference["hidden_bf16"])
        )
    )
    logprobs = _logprobs(model, hidden_full, batch["labels"])
    logprob_ok = bool(
        torch.equal(logprobs.view(torch.int32).cpu(), torch.from_numpy(reference["logprobs"]))
    )
    verdicts["core_hidden"] = hidden_ok
    verdicts["core_logprobs"] = logprob_ok

    # --- padding case: collator cp-multiple padding, real rows must match --
    short = _short_batch(_make_batch(SEQ_LEN, config.vocab_size, device), device)
    padded = _pad_like_collator(short, degree, device)
    assert padded["input_ids"].shape[-1] > SHORT_LEN, "padding case degenerated (no pad added)"
    hidden_padded = _sharded_forward(padded)
    ref_short = torch.from_numpy(reference["hidden_short_bf16"])
    verdicts["padded_hidden"] = bool(
        torch.equal(hidden_padded[:, :SHORT_LEN].contiguous().view(torch.int16).cpu(), ref_short)
    )

    # --- hybrid negative: GDN under Ulysses must RAISE the contract floor --
    hybrid_model, hybrid_config = _build_model(
        ["linear_attention", "full_attention", "linear_attention", "full_attention"], device
    )
    hybrid_batch = _make_batch(SEQ_LEN, hybrid_config.vocab_size, device)
    shard = SEQ_LEN // degree
    hybrid_batch["input_ids"] = hybrid_batch["input_ids"][:, rank * shard : (rank + 1) * shard].contiguous()
    try:
        _forward_hidden(hybrid_model, hybrid_batch)
        verdicts["hybrid_raises"] = False
    except RuntimeError as exc:
        verdicts["hybrid_raises"] = "does not support CP yet" in str(exc)

    gathered_verdicts: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_verdicts, verdicts)
    merged = {}
    for rank_verdicts in gathered_verdicts:
        for name, ok in (rank_verdicts or {}).items():
            merged[name] = merged.get(name, True) and bool(ok)
    if rank == 0:
        for name in sorted(merged):
            print(f"[{'PASS' if merged[name] else 'FAIL'}] ulysses{degree}_{name}", flush=True)
    failed = [name for name, ok in merged.items() if not ok]
    assert not failed, f"Ulysses{degree} byte gate failed: {failed}"
    if rank == 0:
        print(f"Ulysses byte gate passed (degree={degree})", flush=True)


def _pytest_run(degree: int, num_gpus: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "reference.npz")
        env = dict(os.environ)
        env["ULYSSES_GATE_PHASE"] = "ref"
        env["ULYSSES_GATE_REF"] = ref_path
        result = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=900
        )
        assert result.returncode == 0, f"reference phase failed:\n{result.stderr[-2000:]}"
        dist_result = run_distributed_script(
            __file__,
            num_gpus=num_gpus,
            timeout=900,
            extra_env={
                "ULYSSES_GATE_PHASE": "shard",
                "ULYSSES_GATE_DEGREE": str(degree),
                "ULYSSES_GATE_REF": ref_path,
            },
        )
        dist_result.assert_success(f"Ulysses{degree} must be byte-identical to Ulysses1")


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_ulysses2_byte_alignment_exact_program():
        _pytest_run(degree=2, num_gpus=2)

    @skip_if_gpu_count_less_than(4)
    def test_ulysses4_byte_alignment_exact_program():
        # Degree 4 > kv_heads=2: the smallest degree that exercises the GQA
        # KV-replication branch end-to-end (degree 2 == kv_heads skips it).
        _pytest_run(degree=4, num_gpus=4)

    @skip_if_gpu_count_less_than(8)
    def test_ulysses8_byte_alignment_exact_program():
        _pytest_run(degree=8, num_gpus=8)


if __name__ == "__main__":
    phase = os.environ.get("ULYSSES_GATE_PHASE", "ref")
    if phase == "ref":
        _run_reference(os.environ["ULYSSES_GATE_REF"])
    else:
        _run_sharded()
