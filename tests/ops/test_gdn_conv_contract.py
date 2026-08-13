import warnings
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import xorl.ops.linear_attention.layers.gated_deltanet as gated_deltanet
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.modules import ShortConvolution, causal_conv1d_qkv_contract
from xorl.ops.linear_attention.modules.bi_contract import _is_gdn_contract_enabled
from xorl.ops.linear_attention.modules.conv_contract import _pack_conv_weight


KEY_DIM, VALUE_DIM, WIDTH = 2048, 4096, 4


def _make_convs(key_dim: int, value_dim: int, device, dtype, bias: bool = False):
    torch.manual_seed(7)
    convs = []
    for dim in (key_dim, key_dim, value_dim):
        conv = ShortConvolution(hidden_size=dim, kernel_size=WIDTH, bias=bias, activation="silu")
        convs.append(conv.to(device=device, dtype=dtype))
    return convs


def _serving_reference(q_in, k_in, v_in, convs, cu_seqlens=None):
    """The exact serving prefill invocation (gdn_backend.forward_extend)."""
    from xorl.ops.linear_attention.ops.causal_conv1d_triton import causal_conv1d_fn

    x = torch.cat((q_in, k_in, v_in), dim=-1)
    batch, seq_len = x.shape[0], x.shape[1]
    x2d = x.reshape(-1, x.shape[-1])
    if cu_seqlens is None:
        seq_lens = [seq_len] * batch
        qsl = torch.arange(0, (batch + 1) * seq_len, seq_len, device=x.device, dtype=torch.int32)
    else:
        edges = cu_seqlens.tolist()
        seq_lens = [end - start for start, end in zip(edges[:-1], edges[1:], strict=False)]
        qsl = cu_seqlens.to(dtype=torch.int32)
    weight = _pack_conv_weight(*(conv.weight for conv in convs))
    num_seqs = len(seq_lens)
    conv_states = torch.zeros(num_seqs, x2d.shape[1], WIDTH - 1, device=x.device, dtype=x.dtype)
    out = causal_conv1d_fn(
        x2d.transpose(0, 1),
        weight,
        None,
        conv_states=conv_states,
        query_start_loc=qsl,
        seq_lens_cpu=seq_lens,
        cache_indices=torch.arange(num_seqs, device=x.device, dtype=torch.int32),
        has_initial_state=torch.zeros(num_seqs, device=x.device, dtype=torch.bool),
        activation="silu",
    ).transpose(0, 1)
    return out.view_as(x), conv_states


def _tiny_gdn(**overrides) -> GatedDeltaNet:
    kwargs = dict(
        hidden_size=256,
        expand_v=2.0,
        head_dim=64,
        num_heads=2,
        num_v_heads=4,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        conv_size=WIDTH,
        norm_eps=1e-6,
    )
    kwargs.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = GatedDeltaNet(**kwargs)
    layer.train()
    return layer


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestConvContractGPU:
    def test_forward_backward_parity_and_determinism_policy(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(KEY_DIM, VALUE_DIM, device, dtype)
        torch.manual_seed(0)
        q_in = torch.randn(1, 1024, KEY_DIM, device=device, dtype=dtype)
        k_in = torch.randn(1, 1024, KEY_DIM, device=device, dtype=dtype)
        v_in = torch.randn(1, 1024, VALUE_DIM, device=device, dtype=dtype)

        q, k, v = causal_conv1d_qkv_contract(q_in, k_in, v_in, *convs)
        ref, _ = _serving_reference(q_in, k_in, v_in, convs)
        ref_q, ref_k, ref_v = ref.split((KEY_DIM, KEY_DIM, VALUE_DIM), dim=-1)
        assert torch.equal(q, ref_q)
        assert torch.equal(k, ref_k)
        assert torch.equal(v, ref_v)

        self._assert_forward_bitwise_varlen_and_batch()
        self._assert_forward_determinism_double_run()
        self._assert_backward_parity_and_determinism()
        TestConvContractGPU()._assert_end_to_end_gdn_block_policy()

    def _assert_forward_bitwise_varlen_and_batch(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(KEY_DIM, VALUE_DIM, device, dtype)
        torch.manual_seed(1)

        cu_seqlens = torch.tensor([0, 7, 71, 199, 512], device=device, dtype=torch.int32)
        q_in = torch.randn(1, 512, KEY_DIM, device=device, dtype=dtype)
        k_in = torch.randn(1, 512, KEY_DIM, device=device, dtype=dtype)
        v_in = torch.randn(1, 512, VALUE_DIM, device=device, dtype=dtype)
        q, k, v = causal_conv1d_qkv_contract(q_in, k_in, v_in, *convs, cu_seqlens=cu_seqlens)
        ref, _ = _serving_reference(q_in, k_in, v_in, convs, cu_seqlens=cu_seqlens)
        assert torch.equal(torch.cat((q, k, v), dim=-1), ref)

        q_in = torch.randn(3, 128, KEY_DIM, device=device, dtype=dtype)
        k_in = torch.randn(3, 128, KEY_DIM, device=device, dtype=dtype)
        v_in = torch.randn(3, 128, VALUE_DIM, device=device, dtype=dtype)
        q, k, v = causal_conv1d_qkv_contract(q_in, k_in, v_in, *convs)
        ref, _ = _serving_reference(q_in, k_in, v_in, convs)
        assert torch.equal(torch.cat((q, k, v), dim=-1), ref)

    def _assert_forward_determinism_double_run(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(256, 512, device, dtype)
        torch.manual_seed(2)
        inputs = [torch.randn(1, 300, dim, device=device, dtype=dtype) for dim in (256, 256, 512)]
        first = causal_conv1d_qkv_contract(*inputs, *convs)
        second = causal_conv1d_qkv_contract(*inputs, *convs)
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)

    def _assert_backward_parity_and_determinism(self):
        """Same grad_output through both lanes: differences isolate the backward composition."""
        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(256, 512, device, dtype)
        torch.manual_seed(3)
        cu_seqlens = torch.tensor([0, 33, 100, 256], device=device, dtype=torch.int32)
        grad_outs = [torch.randn(1, 256, dim, device=device, dtype=dtype) for dim in (256, 256, 512)]

        def run(fn):
            torch.manual_seed(4)
            inputs = [
                torch.randn(1, 256, dim, device=device, dtype=dtype, requires_grad=True) for dim in (256, 256, 512)
            ]
            outs = fn(inputs)
            leaves = [*inputs, *(conv.weight for conv in convs)]
            return torch.autograd.grad(outs, leaves, grad_outs)

        def eager(inputs):
            return [conv(x, cu_seqlens=cu_seqlens)[0] for x, conv in zip(inputs, convs, strict=True)]

        def contract(inputs):
            return causal_conv1d_qkv_contract(*inputs, *convs, cu_seqlens=cu_seqlens)

        eager_grads = run(eager)
        contract_grads = run(contract)
        for got, ref in zip(contract_grads, eager_grads, strict=True):
            torch.testing.assert_close(got, ref)

        self._assert_backward_determinism_double_run()

    def _assert_backward_determinism_double_run(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(256, 512, device, dtype)

        def run():
            torch.manual_seed(5)
            inputs = [
                torch.randn(2, 96, dim, device=device, dtype=dtype, requires_grad=True) for dim in (256, 256, 512)
            ]
            for conv in convs:
                conv.weight.grad = None
            outs = causal_conv1d_qkv_contract(*inputs, *convs)
            sum(o.float().square().sum() for o in outs).backward()
            return [x.grad.clone() for x in inputs] + [conv.weight.grad.clone() for conv in convs]

        for a, b in zip(run(), run(), strict=True):
            assert torch.equal(a, b)

    def _assert_end_to_end_gdn_block_policy(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        torch.manual_seed(6)
        layer = _tiny_gdn().to(device=device, dtype=dtype)
        hidden = torch.randn(2, 128, 256, device=device, dtype=dtype)

        def run(exact: bool):
            layer.exact_contract = exact
            layer.zero_grad(set_to_none=True)
            out, _, _ = layer(hidden)
            out.float().square().mean().backward()
            return out.detach().clone(), {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}

        eager_out, eager_grads = run(False)
        contract_out, contract_grads = run(True)
        assert eager_grads.keys() == contract_grads.keys()
        assert any(name.startswith("q_conv1d") for name in contract_grads)
        # The two lanes intentionally differ in forward conv bits (~1 bf16 ULP per
        # element); the tolerance bounds that propagation, not backward correctness.
        torch.testing.assert_close(contract_out, eager_out, rtol=5e-2, atol=1e-2)
        for name, ref in eager_grads.items():
            torch.testing.assert_close(contract_grads[name], ref, rtol=5e-2, atol=1e-2, msg=lambda m: f"{name}: {m}")

        self._assert_end_to_end_gdn_block_determinism()

    def _assert_end_to_end_gdn_block_determinism(self):
        device, dtype = torch.device("cuda"), torch.bfloat16
        torch.manual_seed(8)
        layer = _tiny_gdn(exact_contract=True).to(device=device, dtype=dtype)
        hidden = torch.randn(1, 200, 256, device=device, dtype=dtype)
        first, _, _ = layer(hidden)
        second, _, _ = layer(hidden)
        assert torch.equal(first, second)

    def test_matches_sglang_tree_kernel_if_available(self):
        pytest.importorskip("sglang.srt.layers.attention.mamba.causal_conv1d_triton")
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn as sglang_causal_conv1d_fn,
        )

        device, dtype = torch.device("cuda"), torch.bfloat16
        convs = _make_convs(KEY_DIM, VALUE_DIM, device, dtype)
        torch.manual_seed(9)
        q_in = torch.randn(1, 384, KEY_DIM, device=device, dtype=dtype)
        k_in = torch.randn(1, 384, KEY_DIM, device=device, dtype=dtype)
        v_in = torch.randn(1, 384, VALUE_DIM, device=device, dtype=dtype)
        q, k, v = causal_conv1d_qkv_contract(q_in, k_in, v_in, *convs)

        x2d = torch.cat((q_in, k_in, v_in), dim=-1).reshape(384, -1)
        weight = _pack_conv_weight(*(conv.weight for conv in convs))
        out = sglang_causal_conv1d_fn(
            x2d.transpose(0, 1),
            weight,
            None,
            conv_states=torch.zeros(1, x2d.shape[1], WIDTH - 1, device=device, dtype=dtype),
            query_start_loc=torch.tensor([0, 384], device=device, dtype=torch.int32),
            seq_lens_cpu=[384],
            cache_indices=torch.zeros(1, device=device, dtype=torch.int32),
            has_initial_state=torch.zeros(1, device=device, dtype=torch.bool),
            activation="silu",
        ).transpose(0, 1)
        assert torch.equal(torch.cat((q, k, v), dim=-1), out.reshape(1, 384, -1))


@pytest.mark.cpu
class TestConvContractGuards:
    def test_weight_pack_routing_admission_and_state_lifecycle_policy(self, monkeypatch):
        convs = _make_convs(64, 128, torch.device("cpu"), torch.float32)
        packed = _pack_conv_weight(*(conv.weight for conv in convs))
        assert torch.equal(packed[:64], convs[0].weight.squeeze(1))
        assert torch.equal(packed[64:128], convs[1].weight.squeeze(1))
        assert torch.equal(packed[128:], convs[2].weight.squeeze(1))

        with monkeypatch.context() as case_patch:
            self._assert_forward_routes_through_contract_when_armed(case_patch)
        self._assert_contract_admission_policy()
        self._assert_contract_state_lifecycle_policy()

    def _assert_forward_routes_through_contract_when_armed(self, monkeypatch):
        calls = []

        def fake_contract(q_in, k_in, v_in, *convs, cu_seqlens=None, cp_context=None):
            calls.append(cu_seqlens)
            return F.silu(q_in), F.silu(k_in), F.silu(v_in)

        def fake_chunk(**kwargs):
            return kwargs["v"], None

        def fake_gating(A_log, a, b, dt_bias):
            return -A_log.float().exp() * F.softplus(a + dt_bias), torch.sigmoid(b)

        monkeypatch.setattr(gated_deltanet, "causal_conv1d_qkv_contract", fake_contract)
        monkeypatch.setattr(gated_deltanet, "bi_fused_gdn_gating", fake_gating)
        monkeypatch.setattr(gated_deltanet, "chunk_gated_delta_rule", fake_chunk)
        layer = _tiny_gdn(use_gate=False, exact_contract=True)
        out, _, _ = layer(torch.randn(1, 8, 256))
        assert len(calls) == 1
        assert out.shape == (1, 8, 256)

    def _assert_contract_admission_policy(self):
        def use_cache():
            layer = _tiny_gdn(exact_contract=True)
            layer.eval()
            layer(torch.randn(1, 128, 256), use_cache=True)

        def cp_context():
            layer = _tiny_gdn(exact_contract=True)
            context = SimpleNamespace(cu_seqlens=torch.tensor([0, 8]), group=object(), is_first_rank=True)
            layer(torch.randn(1, 8, 256), cp_context=context)

        def no_short_conv():
            _tiny_gdn(use_short_conv=False, exact_contract=True)(torch.randn(1, 8, 256))

        def conv_bias():
            convs = _make_convs(32, 64, torch.device("cpu"), torch.float32, bias=True)
            inputs = [torch.randn(1, 8, dim) for dim in (32, 32, 64)]
            causal_conv1d_qkv_contract(*inputs, *convs)

        cases = [
            ("decode cache", use_cache, RuntimeError, "prefill only"),
            # The blanket "does not support CP" raise is gone: the exact CP
            # program runs under a real FLACPContext. Malformed or duck-typed
            # contexts stay fail-closed at the convolution boundary.
            ("context parallelism, malformed context", cp_context, TypeError, "FLACPContext"),
            ("missing short convolution", no_short_conv, RuntimeError, "requires short convolution"),
            ("convolution bias", conv_bias, NotImplementedError, "bias"),
        ]
        for _label, invoke, error_type, error_pattern in cases:
            with pytest.raises(error_type, match=error_pattern):
                invoke()

    def _assert_contract_state_lifecycle_policy(self):
        seen = []
        exact = _tiny_gdn(exact_contract=True)
        ordinary = _tiny_gdn(exact_contract=False)

        def record(self, hidden_states, **_kwargs):
            seen.append(_is_gdn_contract_enabled())
            return hidden_states, None, None

        exact._forward_impl = MethodType(record, exact)
        ordinary._forward_impl = MethodType(record, ordinary)
        hidden = torch.randn(1, 2, 256)

        exact(hidden)
        ordinary(hidden)
        exact(hidden)

        assert seen == [True, False, True]
        assert not _is_gdn_contract_enabled()

        self._assert_checkpoint_recompute_reestablishes_module_contract()

    def _assert_checkpoint_recompute_reestablishes_module_contract(self):
        seen = []
        exact = _tiny_gdn(exact_contract=True)

        def record(self, hidden_states, **_kwargs):
            seen.append(_is_gdn_contract_enabled())
            return hidden_states.square(), None, None

        exact._forward_impl = MethodType(record, exact)
        hidden = torch.randn(1, 2, 256, requires_grad=True)

        def run(value):
            return exact(value)[0]

        checkpoint(run, hidden, use_reentrant=True).sum().backward()

        assert seen == [True, True]
        assert not _is_gdn_contract_enabled()
