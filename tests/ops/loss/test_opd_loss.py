import pytest
import torch
from safetensors.torch import save_file

from tests._helpers.opd import reference_opd_loss
from tests.ops.loss.conftest import assert_close
from xorl.distillation.teacher_store import TeacherHeadShardView, TeacherHeadStore, prepare_lm_head_teacher_store
from xorl.ops.loss import TokenPartial, opd_loss_function


pytestmark = pytest.mark.cpu


@pytest.fixture
def inputs():
    torch.manual_seed(7)
    batch, seq, vocab, student_h, teacher_h = 2, 5, 13, 6, 8
    hidden_states = torch.randn(batch, seq, student_h) / student_h**0.5
    weight = torch.randn(vocab, student_h) / student_h**0.5
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, 0] = -100
    labels[1, -1] = -100
    teacher_hidden_states = torch.randn(batch, seq, teacher_h) / teacher_h**0.5
    teacher_weight = torch.randn(vocab, teacher_h) / teacher_h**0.5
    teacher_weights = torch.linspace(0.5, 1.5, steps=batch * seq).view(batch, seq)
    return hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights


@pytest.mark.parametrize("num_chunks_case", [1, 2, 7, "n_valid_plus_one", 0])
def test_opd_loss_matches_reference(inputs, num_chunks_case):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    n_valid = int((labels != -100).sum().item())
    num_chunks = n_valid + 1 if num_chunks_case == "n_valid_plus_one" else int(num_chunks_case)
    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        teacher_weights=teacher_weights,
        num_chunks=num_chunks,
    )

    expected = reference_opd_loss(
        hidden_states,
        weight,
        labels,
        teacher_hidden_states,
        teacher_weight,
        teacher_weights,
    )
    assert_close(out.loss, expected)
    assert out.loss.dtype == torch.float32
    assert out.metrics["valid_tokens"] == int((labels != -100).sum().item())


@pytest.mark.parametrize("backend", ["streaming", "tilelang"])
def test_opd_streaming_backends_match_reference(inputs, backend):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    hidden_states = hidden_states.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    teacher_hidden_states = teacher_hidden_states.detach().requires_grad_(True)
    teacher_weight = teacher_weight.detach().requires_grad_(True)

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        teacher_weights=teacher_weights,
        kl_backend=backend,
        vocab_chunk_size=5,
    )

    expected = reference_opd_loss(
        hidden_states,
        weight,
        labels,
        teacher_hidden_states,
        teacher_weight,
        teacher_weights,
    )
    assert_close(out.loss, expected)
    out.loss.backward()
    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    assert weight.grad is not None and weight.grad.isfinite().all()
    assert teacher_hidden_states.grad is None
    assert teacher_weight.grad is None


def test_opd_streaming_lowmem_matches_streaming(inputs):
    """streaming_lowmem must be loss- and gradient-identical to plain streaming.

    The lowmem path keeps lm-head weights in native dtype and upcasts each vocab
    chunk to fp32 in the kernel (no full fp32 copies, no second full grad buffer);
    slicing commutes with the elementwise upcast and vocab chunks partition grad
    rows disjointly, so it is exact -- not an approximation.
    """
    base = [t.clone() for t in inputs[:5]]
    teacher_weights = inputs[5]

    def run(lowmem):
        hs, w, th, tw = (base[0].clone().requires_grad_(True), base[1].clone().requires_grad_(True),
                         base[3].clone().requires_grad_(True), base[4].clone().requires_grad_(True))
        labels = base[2]
        out = opd_loss_function(
            hidden_states=hs, weight=w, labels=labels, teacher_hidden_states=th,
            teacher_lm_head_weight=tw, teacher_weights=teacher_weights,
            kl_backend="streaming", streaming_lowmem=lowmem, vocab_chunk_size=5,
        )
        out.loss.backward()
        return out.loss, w.grad, hs.grad

    loss_a, gw_a, gh_a = run(False)
    loss_b, gw_b, gh_b = run(True)

    expected = reference_opd_loss(base[0], base[1], base[2], base[3], base[4], teacher_weights)
    assert_close(loss_b, expected)
    assert_close(loss_b, loss_a)
    assert_close(gw_b, gw_a)
    assert_close(gh_b, gh_a)


def test_opd_streaming_backend_reads_sharded_teacher_store(inputs, tmp_path):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir()
    save_file({"lm_head.weight": teacher_weight}, str(model_dir / "model.safetensors"))
    manifest = prepare_lm_head_teacher_store(model_dir, tmp_path / "teacher_store", teacher_id=0, shard_rows=4)
    teacher_view = TeacherHeadShardView(
        store=TeacherHeadStore(manifest),
        teacher_id="0",
        device=torch.device("cpu"),
        dtype=None,
    )

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_view,
        teacher_weights=teacher_weights,
        kl_backend="streaming",
        vocab_chunk_size=5,
    )

    expected = reference_opd_loss(hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights)
    assert_close(out.loss, expected)


def test_opd_loss_backward(inputs):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, _ = inputs
    hidden_states = hidden_states.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    teacher_hidden_states = teacher_hidden_states.detach().requires_grad_(True)
    teacher_weight = teacher_weight.detach().requires_grad_(True)

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        num_chunks=2,
    )
    out.loss.backward()

    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    assert weight.grad is not None and weight.grad.isfinite().all()
    assert teacher_hidden_states.grad is None
    assert teacher_weight.grad is None


def test_opd_loss_respects_token_partial_reducer(inputs):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        teacher_weights=teacher_weights,
        num_chunks=2,
        loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
    )

    n_valid = (labels != -100).sum().to(dtype=torch.float32)
    expected = reference_opd_loss(hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights)
    assert_close(out.loss, expected * n_valid)


def test_opd_loss_all_ignored_is_finite(inputs):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, _ = inputs
    labels = torch.full_like(labels, -100)
    hidden_states = hidden_states.to(torch.bfloat16).detach().requires_grad_(True)
    weight = weight.to(torch.bfloat16).detach().requires_grad_(True)

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        lm_head_fp32=True,
    )

    assert out.loss.isfinite()
    assert out.loss.item() == 0.0
    assert out.loss.dtype == torch.float32
    assert out.metrics["valid_tokens"] == 0
    out.loss.backward()
    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    assert weight.grad is not None and weight.grad.isfinite().all()
    assert torch.count_nonzero(hidden_states.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


def test_opd_loss_bf16_inputs_return_fp32_loss(inputs):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    hidden_states = hidden_states.to(torch.bfloat16).detach().requires_grad_(True)
    weight = weight.to(torch.bfloat16).detach().requires_grad_(True)
    teacher_hidden_states = teacher_hidden_states.to(torch.bfloat16)
    teacher_weight = teacher_weight.to(torch.bfloat16)

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        teacher_weights=teacher_weights,
        num_chunks=2,
        lm_head_fp32=True,
        teacher_lm_head_fp32=True,
    )

    assert out.loss.dtype == torch.float32
    out.loss.backward()
    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    assert weight.grad is not None and weight.grad.isfinite().all()


def test_opd_loss_return_per_token(inputs):
    hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights = inputs
    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        teacher_weights=teacher_weights,
        num_chunks=2,
        return_per_token=True,
    )

    assert out.per_token_loss is not None
    assert out.per_token_loss.shape == labels.shape
    assert out.per_token_loss.dtype == torch.float32
    assert torch.count_nonzero(out.per_token_loss[labels == -100]) == 0
    expected = reference_opd_loss(hidden_states, weight, labels, teacher_hidden_states, teacher_weight, teacher_weights)
    denom = (labels != -100).sum().to(dtype=torch.float32)
    assert_close(out.per_token_loss.sum() / denom, expected)


def test_opd_loss_hidden_only_mse_with_zero_kl_weight_returns_hidden_term():
    hidden_states = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]],
        requires_grad=True,
    )
    teacher_hidden_states = torch.tensor([[[1.0, 1.0], [1.0, 3.0], [2.0, 11.0]]])
    labels = torch.tensor([[0, -100, 2]])
    weight = torch.randn(5, 2, requires_grad=True)
    teacher_weight = torch.randn(5, 2)
    hidden_match_weights = torch.tensor([[2.0, 9.0, 0.5]])
    hidden_match_coef = 1.75

    out = opd_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        teacher_hidden_states=teacher_hidden_states,
        teacher_lm_head_weight=teacher_weight,
        hidden_match_weights=hidden_match_weights,
        hidden_match_coef=hidden_match_coef,
        kl_loss_weight=0.0,
        hidden_match_mode="mse",
        kl_backend="streaming",
        vocab_chunk_size=3,
        return_per_token=True,
    )

    valid = labels != -100
    hidden_distance = ((hidden_states.detach() - teacher_hidden_states) ** 2).mean(dim=-1)
    weighted_hidden = hidden_distance[valid] * hidden_match_weights[valid]
    expected_loss = hidden_match_coef * weighted_hidden.sum() / valid.sum()
    expected_per_token = torch.zeros_like(labels, dtype=torch.float32)
    expected_per_token[valid] = hidden_match_coef * weighted_hidden

    assert_close(out.loss, expected_loss)
    torch.testing.assert_close(out.per_token_loss, expected_per_token)
    assert out.metrics["opd_hidden_match_loss"] == pytest.approx(weighted_hidden.mean().item())
    assert out.metrics["opd_hidden_match_raw_loss"] == pytest.approx(hidden_distance[valid].mean().item())
    out.loss.backward()
    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    # kl_loss_weight=0 short-circuit: the KL term is detached from the loss, so its
    # backward through the vocab-parallel LM-head matmul is skipped entirely. The
    # shared LM head `weight` (whose only gradient path here is the KL term) therefore
    # receives no gradient at all. (Before the short-circuit a full KL backward ran
    # just to produce a zero `weight` gradient -- the wasted compute this elides.)
    assert weight.grad is None
    # The KL forward still runs, so the diagnostic metric is preserved for monitoring.
    assert "opd_kl" in out.metrics
