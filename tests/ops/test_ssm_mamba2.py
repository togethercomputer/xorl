"""Parity tests for the Mamba2 (SSD) op and mixer against the HF NemotronHMamba2Mixer torch path.

Oracle notes:
- Single-chunk cases (seq_len <= chunk_size, including padding) are compared strictly against the
  HF `torch_forward` path (mamba_ssm is not installed, so HF falls back to it naturally).
- Multi-chunk cases are compared against a naive sequential SSD recurrence instead: the
  transformers 5.5.3 nemotron_h torch fallback has a bug in its inter-chunk state propagation
  (the original mamba2 code's `decay_chunk.transpose(1, 3)` was lost during restructuring), so it
  drifts from the true SSD recurrence — and from the mamba_ssm kernels — for seq_len > chunk_size.
  `test_hf_torch_path_interchunk_divergence_is_documented` pins this down; if it ever fails,
  transformers fixed the bug and strict multi-chunk HF parity can be re-enabled.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from xorl.ops.ssm import Mamba2Mixer, causal_depthwise_conv1d, ssd_chunked


pytestmark = pytest.mark.cpu

HIDDEN_SIZE = 32
NUM_HEADS = 4
HEAD_DIM = 8
N_GROUPS = 2
STATE_SIZE = 16
CONV_KERNEL = 4


def _build_mixer_pair(chunk_size: int, seed: int = 0) -> tuple[NemotronHMamba2Mixer, Mamba2Mixer]:
    torch.manual_seed(seed)
    config = NemotronHConfig(
        hidden_size=HIDDEN_SIZE,
        mamba_num_heads=NUM_HEADS,
        mamba_head_dim=HEAD_DIM,
        n_groups=N_GROUPS,
        ssm_state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
        use_conv_bias=True,
        chunk_size=chunk_size,
        layers_block_type=["mamba"],
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    hf_mixer = NemotronHMamba2Mixer(config, layer_idx=0).float()
    with torch.no_grad():
        for param in hf_mixer.parameters():
            torch.nn.init.normal_(param, std=0.2)

    xorl_mixer = Mamba2Mixer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        n_groups=N_GROUPS,
        ssm_state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
        use_conv_bias=True,
        chunk_size=chunk_size,
        activation="silu",
        # The HF torch path clamps dt at time_step_min (floor only); align our
        # time_step_limit so the two paths compute identical dt.
        time_step_limit=(config.time_step_min, float("inf")),
        layer_norm_epsilon=config.layer_norm_epsilon,
        use_bias=False,
    ).float()
    # Parameter names match 1:1, so checkpoint loading is a strict pass-through.
    xorl_mixer.load_state_dict(hf_mixer.state_dict(), strict=True)
    return hf_mixer, xorl_mixer


def _ssd_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
) -> torch.Tensor:
    """Naive sequential SSD recurrence (ground truth, matches mamba_ssm kernel semantics)."""
    batch_size, seq_len, num_heads, head_dim = x.shape
    n_groups, state_size = B.shape[2], B.shape[3]
    B = B.repeat_interleave(num_heads // n_groups, dim=2)
    C = C.repeat_interleave(num_heads // n_groups, dim=2)
    state = x.new_zeros(batch_size, num_heads, head_dim, state_size)
    outputs = []
    for t in range(seq_len):
        decay = torch.exp(dt[:, t] * A)  # [batch, heads]
        update = (dt[:, t][..., None, None] * B[:, t][:, :, None, :]) * x[:, t][..., None]
        state = state * decay[..., None, None] + update
        y = (state * C[:, t][:, :, None, :]).sum(-1)
        if D is not None:
            y = y + D[None, :, None] * x[:, t]
        outputs.append(y)
    return torch.stack(outputs, dim=1)


def _random_ssd_inputs(seq_len: int, batch_size: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "x": torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM),
        "dt": torch.rand(batch_size, seq_len, NUM_HEADS) * 0.5 + 0.01,
        "A": -torch.exp(torch.randn(NUM_HEADS) * 0.3),
        "B": torch.randn(batch_size, seq_len, N_GROUPS, STATE_SIZE),
        "C": torch.randn(batch_size, seq_len, N_GROUPS, STATE_SIZE),
        "D": torch.randn(NUM_HEADS),
    }


def _assert_mixer_parity(
    hf_mixer: NemotronHMamba2Mixer,
    xorl_mixer: Mamba2Mixer,
    seq_len: int,
    attention_mask: torch.Tensor | None = None,
    seed: int = 1,
) -> None:
    torch.manual_seed(seed)
    hf_input = torch.randn(2, seq_len, HIDDEN_SIZE, requires_grad=True)
    xorl_input = hf_input.detach().clone().requires_grad_(True)

    hf_out = hf_mixer.torch_forward(hf_input, attention_mask=attention_mask)
    xorl_out, _, _ = xorl_mixer(xorl_input, attention_mask=attention_mask)
    torch.testing.assert_close(xorl_out, hf_out, atol=1e-5, rtol=1e-4)

    hf_out.sum().backward()
    xorl_out.sum().backward()
    torch.testing.assert_close(xorl_input.grad, hf_input.grad, atol=1e-5, rtol=1e-4)

    hf_grads = {name: param.grad for name, param in hf_mixer.named_parameters()}
    xorl_grads = {name: param.grad for name, param in xorl_mixer.named_parameters()}
    assert set(hf_grads) == set(xorl_grads)
    for name, hf_grad in hf_grads.items():
        assert hf_grad is not None, f"HF grad missing for {name}"
        torch.testing.assert_close(xorl_grads[name], hf_grad, atol=1e-5, rtol=1e-4, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("seq_len", [16, 13])
def test_mixer_matches_hf_torch_path_single_chunk(seq_len):
    """Forward + grad parity vs HF, both for seq_len == chunk_size and seq_len % chunk_size != 0."""
    hf_mixer, xorl_mixer = _build_mixer_pair(chunk_size=16)
    _assert_mixer_parity(hf_mixer, xorl_mixer, seq_len=seq_len)


def test_mixer_matches_hf_torch_path_with_attention_mask():
    hf_mixer, xorl_mixer = _build_mixer_pair(chunk_size=16)
    attention_mask = torch.ones(2, 13)
    attention_mask[1, -4:] = 0.0
    _assert_mixer_parity(hf_mixer, xorl_mixer, seq_len=13, attention_mask=attention_mask)


@pytest.mark.parametrize("seq_len", [32, 37])
def test_ssd_chunked_matches_recurrence_multichunk(seq_len):
    """Multi-chunk forward + grads vs the sequential SSD recurrence (chunk_size = 8)."""
    inputs = _random_ssd_inputs(seq_len)
    chunked_inputs = {name: tensor.clone().requires_grad_(True) for name, tensor in inputs.items()}
    reference_inputs = {name: tensor.clone().requires_grad_(True) for name, tensor in inputs.items()}

    y_chunked = ssd_chunked(
        chunked_inputs["x"],
        chunked_inputs["dt"],
        chunked_inputs["A"],
        chunked_inputs["B"],
        chunked_inputs["C"],
        chunked_inputs["D"],
        chunk_size=8,
    )
    y_reference = _ssd_reference(**reference_inputs)
    torch.testing.assert_close(y_chunked, y_reference, atol=1e-4, rtol=1e-4)

    y_chunked.sum().backward()
    y_reference.sum().backward()
    for name in inputs:
        torch.testing.assert_close(
            chunked_inputs[name].grad,
            reference_inputs[name].grad,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m, n=name: f"grad {n}: {m}",
        )


def test_ssd_chunked_no_d_skip():
    inputs = _random_ssd_inputs(seq_len=24, seed=3)
    y_chunked = ssd_chunked(inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"], None, chunk_size=8)
    y_reference = _ssd_reference(inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"], None)
    torch.testing.assert_close(y_chunked, y_reference, atol=1e-4, rtol=1e-4)


def test_mixer_multichunk_matches_sequential_recurrence():
    """Full-mixer multi-chunk check against a reference built from the mixer's own projections."""
    _, xorl_mixer = _build_mixer_pair(chunk_size=8)
    torch.manual_seed(2)
    hidden_states = torch.randn(2, 24, HIDDEN_SIZE)

    xorl_out, _, _ = xorl_mixer(hidden_states)

    with torch.no_grad():
        projected = xorl_mixer.in_proj(hidden_states)
        gate, conv_input, dt = projected.split([xorl_mixer.intermediate_size, xorl_mixer.conv_dim, NUM_HEADS], dim=-1)
        conv_out = F.silu(xorl_mixer.conv1d(conv_input.transpose(1, 2))[..., :24].transpose(1, 2))
        x, b, c = conv_out.split([xorl_mixer.intermediate_size, N_GROUPS * STATE_SIZE, N_GROUPS * STATE_SIZE], dim=-1)
        dt = torch.clamp(F.softplus(dt + xorl_mixer.dt_bias), *xorl_mixer.time_step_limit)
        y = _ssd_reference(
            x.reshape(2, 24, NUM_HEADS, HEAD_DIM),
            dt,
            -torch.exp(xorl_mixer.A_log.float()),
            b.reshape(2, 24, N_GROUPS, STATE_SIZE),
            c.reshape(2, 24, N_GROUPS, STATE_SIZE),
            xorl_mixer.D,
        )
        reference_out = xorl_mixer.out_proj(xorl_mixer.norm(y.reshape(2, 24, -1), gate))

    torch.testing.assert_close(xorl_out, reference_out, atol=1e-5, rtol=1e-4)


def test_hf_torch_path_interchunk_divergence_is_documented():
    """The HF nemotron_h torch fallback mis-propagates state across chunks (transformers 5.5.3).

    If this test starts failing, transformers fixed the bug — switch the multi-chunk tests above
    to strict HF parity.
    """
    hf_mixer, xorl_mixer = _build_mixer_pair(chunk_size=8)
    torch.manual_seed(4)
    hidden_states = torch.randn(2, 24, HIDDEN_SIZE)
    with torch.no_grad():
        hf_out = hf_mixer.torch_forward(hidden_states)
        xorl_out, _, _ = xorl_mixer(hidden_states)
    assert (hf_out - xorl_out).abs().max().item() > 1e-4, (
        "HF torch path now matches the sequential SSD recurrence for multi-chunk inputs; "
        "re-enable strict multi-chunk HF parity."
    )


def test_mixer_bf16_smoke():
    _, xorl_mixer = _build_mixer_pair(chunk_size=8)
    mixer = xorl_mixer.bfloat16()
    hidden_states = torch.randn(2, 24, HIDDEN_SIZE, dtype=torch.bfloat16)
    out, _, _ = mixer(hidden_states)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _seq_idx_from_lengths(lengths: list[int]) -> torch.Tensor:
    return torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)]).unsqueeze(0)


@pytest.mark.parametrize("lengths", [[7, 19, 5], [8, 16, 8]])
def test_ssd_chunked_seq_idx_matches_per_sequence(lengths):
    """Packed SSD with seq_idx vs running each sequence separately (fwd + grads, chunk_size 8).

    [7, 19, 5] puts boundaries inside chunks; [8, 16, 8] aligns them with chunk edges.
    """
    total = sum(lengths)
    inputs = _random_ssd_inputs(total, batch_size=1, seed=7)
    packed_inputs = {name: tensor.clone().requires_grad_(True) for name, tensor in inputs.items()}
    separate_inputs = {name: tensor.clone().requires_grad_(True) for name, tensor in inputs.items()}
    seq_idx = _seq_idx_from_lengths(lengths)

    y_packed = ssd_chunked(
        packed_inputs["x"],
        packed_inputs["dt"],
        packed_inputs["A"],
        packed_inputs["B"],
        packed_inputs["C"],
        packed_inputs["D"],
        chunk_size=8,
        seq_idx=seq_idx,
    )

    pieces = []
    start = 0
    for length in lengths:
        end = start + length
        pieces.append(
            ssd_chunked(
                separate_inputs["x"][:, start:end],
                separate_inputs["dt"][:, start:end],
                separate_inputs["A"],
                separate_inputs["B"][:, start:end],
                separate_inputs["C"][:, start:end],
                separate_inputs["D"],
                chunk_size=8,
            )
        )
        start = end
    y_separate = torch.cat(pieces, dim=1)
    torch.testing.assert_close(y_packed, y_separate, atol=1e-5, rtol=1e-4)

    y_packed.pow(2).sum().backward()
    y_separate.pow(2).sum().backward()
    for name in inputs:
        torch.testing.assert_close(
            packed_inputs[name].grad,
            separate_inputs[name].grad,
            atol=1e-5,
            rtol=1e-4,
            msg=lambda m, n=name: f"grad {n}: {m}",
        )


def test_ssd_chunked_seq_idx_single_sequence_matches_dense():
    """A trivial all-zeros seq_idx must reproduce the dense path exactly."""
    inputs = _random_ssd_inputs(seq_len=24, seed=8)
    seq_idx = torch.zeros(2, 24, dtype=torch.long)
    y_dense = ssd_chunked(inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"], inputs["D"], chunk_size=8)
    y_packed = ssd_chunked(
        inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"], inputs["D"], chunk_size=8, seq_idx=seq_idx
    )
    torch.testing.assert_close(y_packed, y_dense, atol=0.0, rtol=0.0)


def test_causal_conv1d_seq_idx_matches_per_sequence():
    """Boundary-safe depthwise conv vs running each sequence separately (fwd + grads)."""
    torch.manual_seed(9)
    lengths = [7, 19, 5]
    channels = 6
    weight = torch.randn(channels, 1, CONV_KERNEL, requires_grad=True)
    bias = torch.randn(channels, requires_grad=True)
    x = torch.randn(1, sum(lengths), channels, requires_grad=True)
    seq_idx = _seq_idx_from_lengths(lengths)

    y_packed = causal_depthwise_conv1d(x, weight, bias, "silu", seq_idx=seq_idx)

    cu = _cu_seqlens(lengths).tolist()
    y_separate = torch.cat(
        [causal_depthwise_conv1d(x[:, s:e], weight, bias, "silu") for s, e in zip(cu[:-1], cu[1:], strict=False)],
        dim=1,
    )
    torch.testing.assert_close(y_packed, y_separate, atol=1e-6, rtol=1e-5)

    grads_packed = torch.autograd.grad(y_packed.pow(2).sum(), (x, weight, bias), retain_graph=True)
    grads_separate = torch.autograd.grad(y_separate.pow(2).sum(), (x, weight, bias))
    for gp, gs in zip(grads_packed, grads_separate, strict=True):
        torch.testing.assert_close(gp, gs, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("lengths", [[7, 19, 5], [8, 16, 8]])
def test_mixer_packed_varlen_matches_per_sequence(lengths):
    """Gold parity: packed mixer forward with cu_seqlens vs per-sequence runs (fwd + param grads)."""
    _, mixer = _build_mixer_pair(chunk_size=8)
    torch.manual_seed(10)
    total = sum(lengths)
    hidden_states = torch.randn(1, total, HIDDEN_SIZE)
    cu_seqlens = _cu_seqlens(lengths)

    packed_input = hidden_states.clone().requires_grad_(True)
    packed_out, _, _ = mixer(packed_input, cu_seqlens=cu_seqlens)
    packed_out.pow(2).sum().backward()
    packed_grads = {name: param.grad.clone() for name, param in mixer.named_parameters()}
    packed_input_grad = packed_input.grad.clone()
    mixer.zero_grad()

    separate_input = hidden_states.clone().requires_grad_(True)
    cu = cu_seqlens.tolist()
    separate_out = torch.cat([mixer(separate_input[:, s:e])[0] for s, e in zip(cu[:-1], cu[1:], strict=False)], dim=1)
    torch.testing.assert_close(packed_out, separate_out, atol=1e-5, rtol=1e-4)

    separate_out.pow(2).sum().backward()
    torch.testing.assert_close(packed_input_grad, separate_input.grad, atol=1e-5, rtol=1e-4)
    for name, param in mixer.named_parameters():
        torch.testing.assert_close(
            packed_grads[name], param.grad, atol=1e-5, rtol=1e-4, msg=lambda m, n=name: f"grad {n}: {m}"
        )


def test_mixer_packed_full_row_matches_dense():
    """cu_seqlens spanning the whole row must match the dense path (different conv impl, same math)."""
    _, mixer = _build_mixer_pair(chunk_size=8)
    torch.manual_seed(11)
    hidden_states = torch.randn(1, 24, HIDDEN_SIZE)
    with torch.no_grad():
        dense_out, _, _ = mixer(hidden_states)
        packed_out, _, _ = mixer(hidden_states, cu_seqlens=torch.tensor([0, 24], dtype=torch.int32))
    torch.testing.assert_close(packed_out, dense_out, atol=1e-5, rtol=1e-4)


def test_mixer_packed_varlen_rejects_batch_gt_1():
    _, mixer = _build_mixer_pair(chunk_size=8)
    hidden_states = torch.randn(2, 16, HIDDEN_SIZE)
    with pytest.raises(ValueError, match="batch size 1"):
        mixer(hidden_states, cu_seqlens=torch.tensor([0, 8, 16], dtype=torch.int32))


def test_ssd_chunked_use_kernel_unavailable_raises(monkeypatch):
    monkeypatch.setattr("xorl.ops.ssm.ops.ssd.mamba_chunk_scan_combined", None)
    inputs = _random_ssd_inputs(seq_len=8, seed=5)
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        ssd_chunked(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["D"],
            chunk_size=8,
            use_kernel=True,
        )


_KERNEL_AVAILABLE = False
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _kernel  # noqa: F401

    _KERNEL_AVAILABLE = True
except ImportError:
    pass

# The mamba_ssm kernel computes matmuls at tf32-class precision: measured vs the
# fp64 sequential reference, mean relative error ~2e-3 with a ~4e-2 tail (uniform
# across positions, no boundary structure) — tolerances below reflect that.
requires_kernel_gpu = pytest.mark.skipif(
    not (_KERNEL_AVAILABLE and torch.cuda.is_available()),
    reason="requires mamba_ssm and a CUDA device",
)


@pytest.mark.gpu
@requires_kernel_gpu
def test_ssd_chunked_kernel_matches_torch_dense():
    inputs = _random_ssd_inputs(seq_len=37, seed=7)
    kernel_inputs = {k: v.cuda().requires_grad_(True) for k, v in inputs.items()}
    torch_inputs = {k: v.cuda().requires_grad_(True) for k, v in inputs.items()}

    y_kernel = ssd_chunked(
        kernel_inputs["x"],
        kernel_inputs["dt"],
        kernel_inputs["A"],
        kernel_inputs["B"],
        kernel_inputs["C"],
        kernel_inputs["D"],
        chunk_size=8,
        use_kernel=True,
    )
    y_torch = ssd_chunked(
        torch_inputs["x"],
        torch_inputs["dt"],
        torch_inputs["A"],
        torch_inputs["B"],
        torch_inputs["C"],
        torch_inputs["D"],
        chunk_size=8,
        use_kernel=False,
    )
    torch.testing.assert_close(y_kernel.float(), y_torch, atol=5e-2, rtol=5e-2)

    y_kernel.sum().backward()
    y_torch.sum().backward()
    for name in ("x", "dt", "A", "B", "C", "D"):
        torch.testing.assert_close(
            kernel_inputs[name].grad,
            torch_inputs[name].grad,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda m, name=name: f"grad mismatch for {name}: {m}",
        )


@pytest.mark.gpu
@requires_kernel_gpu
def test_ssd_chunked_kernel_matches_torch_packed_seq_idx():
    inputs = _random_ssd_inputs(seq_len=31, batch_size=1, seed=11)
    # Three packed sequences (7, 19, 5) with boundaries inside chunks of 8.
    seq_idx = torch.tensor([[0] * 7 + [1] * 19 + [2] * 5], dtype=torch.int32)

    kernel_inputs = {k: v.cuda().requires_grad_(True) for k, v in inputs.items()}
    torch_inputs = {k: v.cuda().requires_grad_(True) for k, v in inputs.items()}

    y_kernel = ssd_chunked(
        kernel_inputs["x"],
        kernel_inputs["dt"],
        kernel_inputs["A"],
        kernel_inputs["B"],
        kernel_inputs["C"],
        kernel_inputs["D"],
        chunk_size=8,
        seq_idx=seq_idx.cuda(),
        use_kernel=True,
    )
    y_torch = ssd_chunked(
        torch_inputs["x"],
        torch_inputs["dt"],
        torch_inputs["A"],
        torch_inputs["B"],
        torch_inputs["C"],
        torch_inputs["D"],
        chunk_size=8,
        seq_idx=seq_idx.cuda(),
        use_kernel=False,
    )
    torch.testing.assert_close(y_kernel.float(), y_torch, atol=5e-2, rtol=5e-2)

    y_kernel.sum().backward()
    y_torch.sum().backward()
    for name in ("x", "dt", "B", "C"):
        torch.testing.assert_close(
            kernel_inputs[name].grad,
            torch_inputs[name].grad,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda m, name=name: f"grad mismatch for {name}: {m}",
        )
