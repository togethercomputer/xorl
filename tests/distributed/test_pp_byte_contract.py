"""Fail-closed admission tests for the PP byte-boundary contract (CPU-only).

The byte-equality gate itself is tests/distributed/test_pp_byte_alignment.py
(2 GPUs). These tests pin the admission surface: which model families engage,
which cut plans RAISE, and which silent fallbacks became errors.
"""

from __future__ import annotations

import copy
import types
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.pipeline_parallel import (
    _pp_forward,
    _recursive_prune,
    generate_llm_fqn_per_model_part,
)
from xorl.distributed.pp_byte_contract import (
    PPByteContractError,
    engage_pp_byte_contract,
    exact_contract_family,
    validate_pp_exact_microbatch_metadata,
)
from xorl.models.exact_contract import set_glm52_exact_active_lora
from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from xorl.trainers.training_utils import make_pp_loss_fn


def _tiny_model(exact: bool = True) -> Qwen3_5ForCausalLM:
    set_rmsnorm_mode("sglang_fused" if exact else "native")
    config = Qwen3_5Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["full_attention"] * 4,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    # CPU-only tests: keep the Triton fused-SiLU kernel off the forward path.
    config._activation_native = True
    if exact:
        config._qwen35_exact_contract = True
        config._qwen35_rmsnorm_family = "v1"
        # The byte contract fails closed on undeclared dtype AND validates the
        # declaration against actual parameter dtypes, so the exact tiny model
        # must genuinely be bf16.
        config.dtype = torch.bfloat16
        return Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    return Qwen3_5ForCausalLM(config)


def _default_plan(model) -> list:
    pp_config = model.get_pp_module_config()
    return generate_llm_fqn_per_model_part(
        num_stages=2,
        num_layers=pp_config["num_layers"],
        input_fqns=pp_config["input_fqns"],
        layer_prefix=pp_config["layer_prefix"],
        output_fqns=pp_config["output_fqns"],
    )


def _split_parts(model, module_names_per_stage) -> list:
    """Mirror pipeline_module_split's pruning + forward patching (no dist)."""
    parts = []
    for stage_idx, module_names in enumerate(module_names_per_stage):
        part = copy.deepcopy(model)
        fqns_to_keep = set(module_names) | {"model.rotary_emb"}
        _recursive_prune(part, "", fqns_to_keep)
        part._pp_is_first = stage_idx == 0
        part._pp_is_last = stage_idx == len(module_names_per_stage) - 1
        part._pp_stage_idx = stage_idx
        part._pp_original_forward = part.forward
        part.forward = types.MethodType(_pp_forward, part)
        parts.append(part)
    return parts


def _engage(model, plan, parts, *, expects_bf16_mixed_precision: bool = False):
    engage_pp_byte_contract(
        model,
        module_names_per_stage=plan,
        stage_ids=list(range(len(plan))),
        model_parts=parts,
        expects_bf16_mixed_precision=expects_bf16_mixed_precision,
    )


def _valid_metadata_entry(seq_len: int) -> dict:
    return {
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, seq_len], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, seq_len], dtype=torch.int32),
        "max_length_q": seq_len,
        "max_length_k": seq_len,
    }


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------


def test_exact_contract_family_classification():
    assert exact_contract_family(None) is None
    assert exact_contract_family(SimpleNamespace()) is None
    assert exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True)) == "qwen3_5_dense"
    assert (
        exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True, model_type="qwen3_5_moe")) == "qwen3_5_moe"
    )
    assert exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True, num_experts=256)) == "qwen3_5_moe"
    glm_config = SimpleNamespace(_glm52_exact_contract=True)
    assert exact_contract_family(glm_config) == "glm52"
    lora_config = SimpleNamespace()
    set_glm52_exact_active_lora(lora_config, enabled=True)
    assert exact_contract_family(lora_config) == "glm52"


def test_exact_contract_family_prefers_resolution_time_stamp():
    # A present stamp is authoritative, even when legacy flags disagree ...
    assert (
        exact_contract_family(SimpleNamespace(_exact_contract_family="qwen3_5_moe", _qwen35_exact_contract=False))
        == "qwen3_5_moe"
    )
    # ... including a stamped None on a generic model.
    assert exact_contract_family(SimpleNamespace(_exact_contract_family=None, _qwen35_exact_contract=True)) is None
    # Unstamped configs (predating resolution-time stamping) keep the
    # legacy-flag classification through the shared resolver.
    assert exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True)) == "qwen3_5_dense"


# ---------------------------------------------------------------------------
# Engagement and no-op behavior
# ---------------------------------------------------------------------------


def test_generic_models_do_not_engage():
    model = _tiny_model(exact=False)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    assert not any(getattr(part, "_pp_exact_boundary_contract", False) for part in parts)


def test_exact_dense_engages_and_marks_parts():
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    assert all(part._pp_exact_boundary_contract for part in parts)


# ---------------------------------------------------------------------------
# Family floors fail closed
# ---------------------------------------------------------------------------


def test_exact_moe_with_pp_raises():
    stub = SimpleNamespace(config=SimpleNamespace(_qwen35_exact_contract=True, model_type="qwen3_5_moe"))
    with pytest.raises(PPByteContractError, match="Qwen3.5-MoE"):
        engage_pp_byte_contract(stub, module_names_per_stage=[[], []], stage_ids=[0, 1], model_parts=[None, None])


def test_exact_glm52_with_pp_raises():
    stub = SimpleNamespace(config=SimpleNamespace(_glm52_exact_contract=True))
    with pytest.raises(PPByteContractError, match="GLM-5.2"):
        engage_pp_byte_contract(stub, module_names_per_stage=[[], []], stage_ids=[0, 1], model_parts=[None, None])


# ---------------------------------------------------------------------------
# Cut-point admission fails closed
# ---------------------------------------------------------------------------


def test_output_modules_split_off_the_last_stage_raise():
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    # Move the final norm onto stage 0: the final norm and head form one
    # rounding boundary with the layer stack and may not be separated.
    plan = [list(plan[0]) + ["model.norm"], [name for name in plan[1] if name != "model.norm"]]
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="output module"):
        _engage(model, plan, parts)


def test_non_contiguous_stage_layers_raise():
    model = _tiny_model(exact=True)
    plan = [
        ["model.embed_tokens", "model.layers.0", "model.layers.2"],
        ["model.layers.1", "model.layers.3", "model.norm", "lm_head"],
    ]
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="not contiguous"):
        _engage(model, plan, parts)


def test_descending_stage_order_raises():
    model = _tiny_model(exact=True)
    plan = [
        ["model.embed_tokens", "model.layers.2", "model.layers.3"],
        ["model.layers.0", "model.layers.1", "model.norm", "lm_head"],
    ]
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="ascending stage order"):
        _engage(model, plan, parts)


def test_stage_without_decoder_layers_raises():
    model = _tiny_model(exact=True)
    plan = [
        ["model.embed_tokens"],
        ["model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3", "model.norm", "lm_head"],
    ]
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="owns no decoder layers"):
        _engage(model, plan, parts)


def test_non_bf16_model_dtype_raises():
    model = _tiny_model(exact=True)
    model.config.dtype = torch.float16
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="not admitted"):
        _engage(model, plan, parts)


def test_undeclared_model_dtype_raises():
    """Fail closed on ABSENCE: an ordinary model with no dtype declaration
    (e.g. a default FP32 construction) must never be markable as exact."""
    model = _tiny_model(exact=True)
    model.config.dtype = None
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="declares no weight dtype"):
        _engage(model, plan, parts)


def test_declared_bf16_with_fp32_params_raises():
    """The declaration must match the RESOLVED reality: a model declaring
    bfloat16 while its parameters are float32 (and no bf16 mixed-precision
    compute policy is coming) must not be marked exact."""
    model = _tiny_model(exact=True).to(torch.float32)
    assert model.config.dtype == torch.bfloat16  # declared bf16, actual fp32
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    with pytest.raises(PPByteContractError, match="declared=bfloat16, actual=float32"):
        _engage(model, plan, parts)
    assert not any(getattr(part, "_pp_exact_boundary_contract", False) for part in parts)


def test_fp32_masters_admitted_only_with_mixed_precision_intent():
    """Uniform fp32 masters are the production full-weight shape; they are
    admitted only when the caller declares bf16 mixed-precision compute (the
    runtime wire assertions then verify the actual bytes)."""
    model = _tiny_model(exact=True).to(torch.float32)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts, expects_bf16_mixed_precision=True)
    assert all(part._pp_exact_boundary_contract for part in parts)


def test_rogue_fp32_param_among_bf16_raises():
    """An arbitrary bf16/fp32 mixture is not admitted: any fp32 parameter that
    is not an APPROVED pin RAISES naming the parameter."""
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    rogue = parts[1].model.layers[2].mlp.down_proj.weight
    rogue.data = rogue.data.to(torch.float32)
    with pytest.raises(PPByteContractError, match=r"down_proj\.weight.*not an approved fp32 pin"):
        _engage(model, plan, parts)
    assert not any(getattr(part, "_pp_exact_boundary_contract", False) for part in parts)


def test_approved_gdn_fp32_pins_admitted():
    """The exact GDN's fp32-pinned gating parameters (A_log/dt_bias) are the
    approved mixture: a bf16 model containing them engages and is marked."""
    set_rmsnorm_mode("sglang_fused")
    config = Qwen3_5Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"] * 2,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._activation_native = True
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v1"
    config.dtype = torch.bfloat16
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    # The GDN pins survive .to(bf16) by design — assert the premise holds.
    pinned = {n for n, p in model.named_parameters() if p.dtype == torch.float32}
    assert pinned and all(n.endswith(("A_log", "dt_bias")) for n in pinned)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    assert all(part._pp_exact_boundary_contract for part in parts)


def test_non_bf16_wire_input_raises_on_marked_stage():
    """The received inter-stage tensor is the wire reality; a marked non-first
    stage must refuse anything that is not bf16 regardless of declarations."""
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    part = parts[1]
    assert not part._pp_is_first
    seq_len = 8
    part._pp_batch_metadata = deque([_valid_metadata_entry(seq_len)])
    fp32_hidden = torch.zeros(1, seq_len, model.config.hidden_size, dtype=torch.float32)
    with torch.enable_grad():
        with pytest.raises(PPByteContractError, match="received inter-stage hidden state"):
            part.forward(fp32_hidden)


# ---------------------------------------------------------------------------
# Global layer identity fails closed
# ---------------------------------------------------------------------------


def test_reindexed_layer_container_raises():
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    # Simulate Megatron-style local re-indexing on the second stage: kept
    # layers compacted to local indices 0..n-1.
    container = parts[1].model.layers
    kept = [layer for layer in container if layer is not None]
    container._modules = {str(i): layer for i, layer in enumerate(kept)}
    with pytest.raises(PPByteContractError, match="index preservation"):
        _engage(model, plan, parts)


def test_wrong_layer_idx_raises():
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    # The first layer of stage 1 claims to be layer 0: the input-norm kernel
    # family would flip from the residual tree to the layer-0 program.
    parts[1].model.layers[2].layer_idx = 0
    with pytest.raises(PPByteContractError, match="layer_idx"):
        _engage(model, plan, parts)


# ---------------------------------------------------------------------------
# Metadata and head-program admission fail closed
# ---------------------------------------------------------------------------


def test_marked_part_raises_on_metadata_starvation():
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    part = parts[0]
    part._pp_batch_metadata = deque()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.enable_grad():
        with pytest.raises(PPByteContractError, match="position_ids"):
            part.forward(input_ids)


def test_marked_part_raises_on_incomplete_varlen_metadata():
    """Partial metadata must RAISE, not silently degrade: a queue entry with
    position_ids but no cu_seq_lens_* would let the attention entry treat a
    packed batch as ONE document (cross-document attention — silent
    numerics)."""
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    part = parts[0]
    seq_len = 8
    entry = _valid_metadata_entry(seq_len)
    del entry["cu_seq_lens_q"]  # deliberately absent
    part._pp_batch_metadata = deque([entry])
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    with torch.enable_grad():
        with pytest.raises(PPByteContractError, match="cu_seq_lens_q"):
            part.forward(input_ids)


@pytest.mark.parametrize(
    ("mutation", "expected_match"),
    [
        # Present-but-None passed the old presence check and still reached the
        # single-document fallback.
        ({"cu_seq_lens_q": None}, "got NoneType"),
        ({"cu_seq_lens_q": torch.tensor([], dtype=torch.int32)}, ">= 2 boundaries"),
        ({"cu_seq_lens_q": torch.tensor([0, 5, 3, 8], dtype=torch.int32)}, "strictly increasing"),
        ({"cu_seq_lens_q": torch.tensor([0, 7], dtype=torch.int32)}, "last boundary"),
        ({"cu_seq_lens_q": torch.tensor([0, 8], dtype=torch.int64)}, "expected int32"),
        ({"cu_seq_lens_k": torch.tensor([0, 4, 8], dtype=torch.int32)}, "spans differ"),
        ({"max_length_q": 3}, "kernel undersizing"),
    ],
)
def test_marked_part_raises_on_malformed_varlen_metadata(mutation, expected_match):
    """Value validation, not key presence: None values, empty tensors,
    non-monotonic boundaries, wrong endpoints, dtype drift, q/k span
    mismatches, and undersized max lengths all RAISE before any compute."""
    model = _tiny_model(exact=True)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    part = parts[0]
    seq_len = 8
    entry = _valid_metadata_entry(seq_len)
    entry.update(mutation)
    part._pp_batch_metadata = deque([entry])
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    with torch.enable_grad():
        with pytest.raises(PPByteContractError, match=expected_match):
            part.forward(input_ids)


def test_valid_varlen_metadata_passes_value_validation():
    seq_len = 8
    entry = _valid_metadata_entry(seq_len)
    position_ids = entry.pop("position_ids")
    x = torch.zeros(1, seq_len, dtype=torch.long)
    validate_pp_exact_microbatch_metadata(x, position_ids, entry)
    # Packed two-document form is also valid.
    entry["cu_seq_lens_q"] = torch.tensor([0, 3, seq_len], dtype=torch.int32)
    entry["cu_seq_lens_k"] = torch.tensor([0, 3, seq_len], dtype=torch.int32)
    validate_pp_exact_microbatch_metadata(x, position_ids, entry)


def test_generic_part_keeps_silent_fallback():
    model = _tiny_model(exact=False)
    plan = _default_plan(model)
    parts = _split_parts(model, plan)
    _engage(model, plan, parts)
    part = parts[0]
    part._pp_batch_metadata = deque()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        hidden = part.forward(input_ids)
    assert hidden.shape == (1, 8, model.config.hidden_size)


def test_bi_fused_pp_training_loss_raises():
    with pytest.raises(NotImplementedError, match="bi_fused"):
        make_pp_loss_fn("bi_fused")
