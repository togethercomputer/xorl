from types import SimpleNamespace

import torch

import xorl.distributed.parallel_state as parallel_state_module
from xorl.models.layers.moe import MoEBlock, MoEExperts
from xorl.ops.moe.activations import apply_moe_activation


def _fake_tp_state(tp_size: int = 2):
    return SimpleNamespace(ep_enabled=False, tp_size=tp_size)


def test_sglang_moe_tp_sim_env_enables_no_ep_tp1(monkeypatch):
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    tp1_state = _fake_tp_state(tp_size=1)

    monkeypatch.delenv("XORL_SGLANG_MOE_TP_SIM", raising=False)
    assert not experts.sglang_moe_tp_sim_enabled(tp1_state)

    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "sglang_runner")
    assert experts.sglang_moe_tp_sim_enabled(tp1_state)

    ep_state = SimpleNamespace(ep_enabled=True, tp_size=1)
    assert not experts.sglang_moe_tp_sim_enabled(ep_state)


def test_sglang_moe_tp_sim_layer_filter(monkeypatch):
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    tp1_state = _fake_tp_state(tp_size=1)
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "sglang")
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM_LAYERS", "2,4")

    assert not experts.sglang_moe_tp_sim_enabled(tp1_state)

    experts.layer_idx = 3
    assert not experts.sglang_moe_tp_sim_enabled(tp1_state)

    experts.layer_idx = 4
    assert experts.sglang_moe_tp_sim_enabled(tp1_state)


def _manual_sglang_tp_sim_direct(
    experts: MoEExperts,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
    routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
    output = hidden_flat.new_zeros(hidden_flat.shape)
    shard_intermediate = experts.intermediate_size // tp_size

    for tp_rank in range(tp_size):
        start = tp_rank * shard_intermediate
        end = start + shard_intermediate
        shard_output = hidden_flat.new_zeros(hidden_flat.shape)
        for expert_idx in range(experts.num_experts):
            mask = selected_flat == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            tokens = hidden_flat.index_select(0, token_rows)
            gate = tokens.matmul(experts.gate_up_proj[expert_idx, :, start:end])
            up = tokens.matmul(
                experts.gate_up_proj[
                    expert_idx,
                    :,
                    experts.intermediate_size + start : experts.intermediate_size + end,
                ]
            )
            activated = apply_moe_activation(experts.hidden_act, gate, up)
            expert_out = activated.matmul(experts.down_proj[expert_idx, start:end, :])
            expert_out = expert_out * routing_flat[token_rows, topk_slots].unsqueeze(-1)
            shard_output.index_add_(0, token_rows, expert_out)
        output = output + shard_output

    return output.reshape(hidden_states.shape)


def _manual_sglang_tp_sim_direct_bf16_reduce(
    experts: MoEExperts,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
    routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
    output = hidden_flat.new_zeros(hidden_flat.shape)
    shard_intermediate = experts.intermediate_size // tp_size

    for tp_rank in range(tp_size):
        start = tp_rank * shard_intermediate
        end = start + shard_intermediate
        shard_output = hidden_flat.new_zeros(hidden_flat.shape)
        for expert_idx in range(experts.num_experts):
            mask = selected_flat == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            tokens = hidden_flat.index_select(0, token_rows)
            gate = tokens.matmul(experts.gate_up_proj[expert_idx, :, start:end])
            up = tokens.matmul(
                experts.gate_up_proj[
                    expert_idx,
                    :,
                    experts.intermediate_size + start : experts.intermediate_size + end,
                ]
            )
            activated = apply_moe_activation(experts.hidden_act, gate, up)
            expert_out = activated.matmul(experts.down_proj[expert_idx, start:end, :])
            expert_out = expert_out * routing_flat[token_rows, topk_slots].unsqueeze(-1)
            shard_output.index_add_(0, token_rows, expert_out)
        output = output.to(torch.bfloat16) + shard_output.to(torch.bfloat16)

    return output.reshape(hidden_states.shape)


def _manual_sglang_tp_sim_cache(
    experts: MoEExperts,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    selected_flat = selected_experts.reshape(hidden_flat.shape[0], -1)
    routing_flat = routing_weights.reshape(hidden_flat.shape[0], -1)
    output = hidden_flat.new_zeros(hidden_flat.shape)
    shard_intermediate = experts.intermediate_size // tp_size

    for tp_rank in range(tp_size):
        start = tp_rank * shard_intermediate
        end = start + shard_intermediate
        topk = selected_flat.shape[1]
        gate_up_cache = hidden_flat.new_zeros(hidden_flat.shape[0] * topk, 2 * shard_intermediate)
        down_cache = hidden_flat.new_zeros(hidden_flat.shape[0], topk, hidden_flat.shape[-1])

        for expert_idx in range(experts.num_experts):
            mask = selected_flat == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            assignment_rows = token_rows * topk + topk_slots
            tokens = hidden_flat.index_select(0, token_rows)
            gate = tokens.matmul(experts.gate_up_proj[expert_idx, :, start:end])
            up = tokens.matmul(
                experts.gate_up_proj[
                    expert_idx,
                    :,
                    experts.intermediate_size + start : experts.intermediate_size + end,
                ]
            )
            gate_up_cache[assignment_rows, :shard_intermediate] = gate
            gate_up_cache[assignment_rows, shard_intermediate:] = up

        gate = gate_up_cache[:, :shard_intermediate]
        up = gate_up_cache[:, shard_intermediate:]
        activated_cache = apply_moe_activation(experts.hidden_act, gate, up)

        for expert_idx in range(experts.num_experts):
            mask = selected_flat == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            assignment_rows = token_rows * topk + topk_slots
            activated = activated_cache.index_select(0, assignment_rows)
            expert_out = activated.matmul(experts.down_proj[expert_idx, start:end, :])
            expert_out = expert_out * routing_flat[token_rows, topk_slots].unsqueeze(-1)
            down_cache[token_rows, topk_slots, :] = expert_out
        shard_output = down_cache.to(torch.float32).sum(dim=1).to(down_cache.dtype)
        output = output + shard_output

    return output.reshape(hidden_states.shape)


def _patch_fake_group_gemm(monkeypatch):
    import xorl.ops.group_gemm.kernel.group_gemm as group_gemm  # noqa: PLC0415
    import xorl.ops.group_gemm.kernel.moe as moe_kernel  # noqa: PLC0415

    def fake_expert_histogram(expert_index, num_experts):
        return torch.bincount(expert_index.reshape(-1), minlength=num_experts).to(torch.int32)

    def fake_moe_index_compute(expert_index, cumsum_t):
        starts = torch.cat([cumsum_t.new_zeros(1), cumsum_t[:-1]]).to(torch.long)
        offsets = starts.clone()
        scatter_index = torch.empty_like(expert_index, dtype=torch.long)
        for token_idx in range(expert_index.shape[0]):
            for topk_idx in range(expert_index.shape[1]):
                expert_idx = int(expert_index[token_idx, topk_idx].item())
                scatter_index[token_idx, topk_idx] = offsets[expert_idx]
                offsets[expert_idx] += 1
        return scatter_index

    def fake_moe_scatter(hidden_states, scatter_index):
        output = hidden_states.new_empty(scatter_index.numel(), hidden_states.shape[-1])
        for token_idx in range(scatter_index.shape[0]):
            for topk_idx in range(scatter_index.shape[1]):
                output[int(scatter_index[token_idx, topk_idx].item())] = hidden_states[token_idx]
        return output

    def fake_group_gemm_same_nk(*, a, b, cumsum_M, max_M, transpose_b=False, **kwargs):
        del kwargs
        output_dim = b.shape[1] if transpose_b else b.shape[2]
        output = a.new_empty(max_M, output_dim)
        start = 0
        for expert_idx, end_value in enumerate(cumsum_M.tolist()):
            end = int(end_value)
            if end > start:
                weight = b[expert_idx].transpose(0, 1) if transpose_b else b[expert_idx]
                output[start:end] = a[start:end].matmul(weight)
            start = end
        return output

    monkeypatch.setattr(moe_kernel, "expert_histogram", fake_expert_histogram)
    monkeypatch.setattr(moe_kernel, "moe_index_compute", fake_moe_index_compute)
    monkeypatch.setattr(moe_kernel, "moe_scatter", fake_moe_scatter)
    monkeypatch.setattr(group_gemm, "group_gemm_same_nk", fake_group_gemm_same_nk)


def test_sglang_moe_tp_sim_matches_manual_shard_reduce(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "1")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_direct(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_sim_bf16_reduce_forces_shard_accumulation_dtype(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "1")
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM_BF16_REDUCE", "1")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_direct_bf16_reduce(
        experts,
        hidden_states,
        routing_weights,
        selected_experts,
        tp_size=2,
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_cache_mode_matches_manual_cache_reduce(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "cache")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_cache(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_sim_size_override_under_tp1(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "cache")
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM_SIZE", "2")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=1))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_cache(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_triton_mode_shards_backend_call(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "triton")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    calls = []

    def fake_triton_moe_forward(
        *,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj,
        hidden_act,
        swiglu_limit,
        **kwargs,
    ):
        del kwargs, gate_proj, up_proj, swiglu_limit
        calls.append(
            {
                "gate_up_shape": tuple(gate_up_proj.shape),
                "down_shape": tuple(down_proj.shape),
                "hidden_shape": tuple(hidden_states.shape),
            }
        )
        shard_intermediate = gate_up_proj.shape[-1] // 2
        output = hidden_states.new_zeros(hidden_states.shape)
        for expert_idx in range(num_experts):
            mask = selected_experts == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            tokens = hidden_states.index_select(0, token_rows)
            gate_up = tokens.matmul(gate_up_proj[expert_idx])
            gate, up = gate_up.split(shard_intermediate, dim=-1)
            activated = apply_moe_activation(hidden_act, gate, up)
            expert_out = activated.matmul(down_proj[expert_idx])
            expert_out = expert_out * routing_weights[token_rows, topk_slots].unsqueeze(-1)
            output.index_add_(0, token_rows, expert_out)
        return output

    import xorl.ops.moe.triton as triton_moe  # noqa: PLC0415

    monkeypatch.setattr(triton_moe, "triton_moe_forward", fake_triton_moe_forward)

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_direct(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)
    assert calls == [
        {"gate_up_shape": (3, 4, 6), "down_shape": (3, 3, 4), "hidden_shape": (5, 4)},
        {"gate_up_shape": (3, 4, 6), "down_shape": (3, 3, 4), "hidden_shape": (5, 4)},
    ]


def test_sglang_moe_tp_triton_sgl_reduce_mode_matches_manual_cache_reduce(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "triton_sgl_reduce")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))
    _patch_fake_group_gemm(monkeypatch)

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_cache(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_deep_gemm_mode_matches_manual_cache_reduce(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "deep_gemm")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))
    _patch_fake_group_gemm(monkeypatch)

    calls = []

    def fake_deep_gemm_group_gemm_same_nk(*, a, b, cumsum_M):
        calls.append({"a_shape": tuple(a.shape), "b_shape": tuple(b.shape), "cumsum": tuple(cumsum_M.tolist())})
        output = a.new_empty(a.shape[0], b.shape[2])
        start = 0
        for expert_idx, end_value in enumerate(cumsum_M.tolist()):
            end = int(end_value)
            if end > start:
                output[start:end] = a[start:end].matmul(b[expert_idx])
            start = end
        return output

    monkeypatch.setattr(
        MoEExperts,
        "_deep_gemm_group_gemm_same_nk",
        staticmethod(fake_deep_gemm_group_gemm_same_nk),
    )

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_cache(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)
    assert calls == [
        {"a_shape": (10, 4), "b_shape": (3, 4, 6), "cumsum": (3, 6, 10)},
        {"a_shape": (10, 3), "b_shape": (3, 3, 4), "cumsum": (3, 6, 10)},
        {"a_shape": (10, 4), "b_shape": (3, 4, 6), "cumsum": (3, 6, 10)},
        {"a_shape": (10, 3), "b_shape": (3, 3, 4), "cumsum": (3, 6, 10)},
    ]


def test_sglang_moe_tp_sglang_mode_uses_sglang_weight_layout(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "sglang")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    calls = []

    def fake_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        *,
        activation,
        filter_expert,
        **kwargs,
    ):
        calls.append(
            {
                "w1_shape": tuple(w1.shape),
                "w2_shape": tuple(w2.shape),
                "activation": activation,
                "gemm1_limit": kwargs["gemm1_limit"],
                "filter_expert": filter_expert,
            }
        )
        shard_intermediate = w1.shape[1] // 2
        output = hidden_states.new_zeros(hidden_states.shape)
        for expert_idx in range(w1.shape[0]):
            mask = topk_ids == expert_idx
            if not bool(mask.any().item()):
                continue
            token_rows, topk_slots = mask.nonzero(as_tuple=True)
            tokens = hidden_states.index_select(0, token_rows)
            gate_up = tokens.matmul(w1[expert_idx].transpose(0, 1))
            gate, up = gate_up.split(shard_intermediate, dim=-1)
            activated = apply_moe_activation(activation, gate, up)
            expert_out = activated.matmul(w2[expert_idx].transpose(0, 1))
            expert_out = expert_out * topk_weights[token_rows, topk_slots].unsqueeze(-1)
            output.index_add_(0, token_rows, expert_out)
        return output

    monkeypatch.setattr(
        MoEExperts,
        "_load_sglang_fused_experts_impl",
        staticmethod(lambda: fake_fused_experts_impl),
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_direct(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)
    assert calls == [
        {
            "w1_shape": (3, 6, 4),
            "w2_shape": (3, 4, 3),
            "activation": "silu",
            "gemm1_limit": None,
            "filter_expert": False,
        },
        {
            "w1_shape": (3, 6, 4),
            "w2_shape": (3, 4, 3),
            "activation": "silu",
            "gemm1_limit": None,
            "filter_expert": False,
        },
    ]


def test_sglang_moe_tp_sglang_runner_mode_uses_runner_contract(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "sglang_runner")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(0)
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=6, moe_implementation="eager")
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn_like(experts.gate_up_proj) * 0.1)
        experts.down_proj.copy_(torch.randn_like(experts.down_proj) * 0.1)

    hidden_states = torch.randn(5, 4)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.80, 0.20],
            [0.50, 0.50],
        ],
        dtype=hidden_states.dtype,
    )

    calls = []

    class FakeMoeRunnerBackend:
        TRITON = "triton"

    class FakeMoeRunnerConfig(SimpleNamespace):
        pass

    class FakeTritonMoeQuantInfo(SimpleNamespace):
        pass

    class FakeStandardTopKOutput(SimpleNamespace):
        pass

    class FakeStandardDispatchOutput(SimpleNamespace):
        pass

    class FakeMoeRunner:
        def __init__(self, backend, config):
            calls.append(
                {
                    "backend": backend,
                    "activation": config.activation,
                    "hidden_size": config.hidden_size,
                    "intermediate_size_per_partition": config.intermediate_size_per_partition,
                    "top_k": config.top_k,
                    "inplace": config.inplace,
                    "gemm1_clamp_limit": config.gemm1_clamp_limit,
                    "gate_up_interleaved": config.gate_up_interleaved,
                }
            )

        def run(self, dispatch_output, quant_info):
            hidden = dispatch_output.hidden_states
            topk_weights = dispatch_output.topk_output.topk_weights
            topk_ids = dispatch_output.topk_output.topk_ids
            w13 = quant_info.w13_weight
            w2 = quant_info.w2_weight
            shard_intermediate = w13.shape[1] // 2
            output = hidden.new_zeros(hidden.shape)
            for expert_idx in range(w13.shape[0]):
                mask = topk_ids == expert_idx
                if not bool(mask.any().item()):
                    continue
                token_rows, topk_slots = mask.nonzero(as_tuple=True)
                tokens = hidden.index_select(0, token_rows)
                gate_up = tokens.matmul(w13[expert_idx].transpose(0, 1))
                gate, up = gate_up.split(shard_intermediate, dim=-1)
                activated = apply_moe_activation("silu", gate, up)
                expert_out = activated.matmul(w2[expert_idx].transpose(0, 1))
                expert_out = expert_out * topk_weights[token_rows, topk_slots].unsqueeze(-1)
                output.index_add_(0, token_rows, expert_out)
            return SimpleNamespace(hidden_states=output)

    monkeypatch.setattr(
        MoEExperts,
        "_load_sglang_moe_runner_stack",
        staticmethod(
            lambda: (
                FakeMoeRunner,
                FakeMoeRunnerBackend,
                FakeMoeRunnerConfig,
                FakeTritonMoeQuantInfo,
                FakeStandardDispatchOutput,
                FakeStandardTopKOutput,
            )
        ),
    )

    actual = experts(hidden_states, routing_weights, selected_experts)
    expected = _manual_sglang_tp_sim_direct(experts, hidden_states, routing_weights, selected_experts, tp_size=2)

    torch.testing.assert_close(actual, expected)
    assert calls == [
        {
            "backend": "triton",
            "activation": "silu",
            "hidden_size": 4,
            "intermediate_size_per_partition": 3,
            "top_k": 2,
            "inplace": False,
            "gemm1_clamp_limit": None,
            "gate_up_interleaved": False,
        },
        {
            "backend": "triton",
            "activation": "silu",
            "hidden_size": 4,
            "intermediate_size_per_partition": 3,
            "top_k": 2,
            "inplace": False,
            "gemm1_clamp_limit": None,
            "gate_up_interleaved": False,
        },
    ]


def test_eager_moe_block_uses_tp_sim_without_per_expert_bypass(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "1")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(1)
    block = MoEBlock(
        hidden_size=4,
        num_experts=3,
        top_k=2,
        intermediate_size=6,
        moe_implementation="eager",
    )
    with torch.no_grad():
        block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj) * 0.1)
        block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj) * 0.1)
    hidden_states = torch.randn(2, 3, 4)

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    routing_weights, selected_experts, _ = block.route(flat_hidden)
    expected = _manual_sglang_tp_sim_direct(
        block.experts,
        flat_hidden,
        routing_weights,
        selected_experts,
        tp_size=2,
    ).reshape(hidden_states.shape)

    actual, _ = block(hidden_states)

    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_sim_carry_shards_survives_moe_block_reshape(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "1")
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM_CARRY_SHARDS", "1")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(2)
    block = MoEBlock(
        hidden_size=4,
        num_experts=3,
        top_k=2,
        intermediate_size=6,
        moe_implementation="eager",
    )
    with torch.no_grad():
        block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj) * 0.1)
        block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj) * 0.1)
    hidden_states = torch.randn(2, 3, 4)

    actual, _ = block(hidden_states)
    carried_shards = getattr(actual, "_xorl_sglang_moe_tp_shards", None)

    assert carried_shards is not None
    assert len(carried_shards) == 2
    assert all(shard.shape == actual.shape for shard in carried_shards)
    expected = carried_shards[0] + carried_shards[1]
    torch.testing.assert_close(actual, expected)


def test_sglang_moe_tp_sim_captures_flat_shards(monkeypatch):
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM", "1")
    monkeypatch.setenv("XORL_SGLANG_MOE_TP_SIM_CARRY_SHARDS", "1")
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_tp_state(tp_size=2))

    torch.manual_seed(3)
    block = MoEBlock(
        hidden_size=4,
        num_experts=3,
        top_k=2,
        intermediate_size=6,
        moe_implementation="eager",
    )
    with torch.no_grad():
        block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj) * 0.1)
        block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj) * 0.1)
    captures = {}
    block._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())
    hidden_states = torch.randn(2, 3, 4)

    actual, _ = block(hidden_states)

    assert "moe_experts_output_tp_shard_0" in captures
    assert "moe_experts_output_tp_shard_1" in captures
    assert "moe_experts_output_tp_shard_sum" in captures
    assert captures["moe_experts_output_tp_shard_0"].shape == (6, 4)
    assert captures["moe_experts_output_tp_shard_1"].shape == (6, 4)
    torch.testing.assert_close(
        captures["moe_experts_output_tp_shard_sum"],
        captures["moe_experts_output_tp_shard_0"] + captures["moe_experts_output_tp_shard_1"],
    )
    torch.testing.assert_close(captures["moe_experts_output_tp_shard_sum"], captures["moe_experts_output"])
    torch.testing.assert_close(actual, captures["moe_experts_output"].reshape_as(actual))
