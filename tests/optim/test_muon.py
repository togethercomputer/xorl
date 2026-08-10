from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xorl.optim.gram_newton_schulz as gram_newton_schulz
import xorl.optim.muon as muon_module
from tests.models.test_deepseek_v3_model import _tiny_config as _deepseek_v3_tiny_config
from tests.models.test_nemotron_h_model import _build_model as _build_nemotron_h_model
from xorl.models.layers.moe import MoEExperts
from xorl.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from xorl.optim import Muon, build_optimizer
from xorl.optim.gram_newton_schulz import GramNewtonSchulzOrthogonalizer, expand_ns_coefficients, find_best_restarts
from xorl.optim.optimizer import _collect_fused_gate_up_ids


pytestmark = [pytest.mark.cpu]


def _build_trainer_style_muon(model: nn.Module) -> Muon:
    """Construct Muon the way Trainer._build_optimizer / the server runner do."""
    optimizer = build_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
        optimizer_type="muon",
        optimizer_kwargs={"muon_ns_use_quack_kernels": False},
    )
    assert isinstance(optimizer, Muon)
    return optimizer


def _split_group_ids(optimizer: Muon) -> tuple[set, set, set]:
    muon_ids: set = set()
    adamw_ids: set = set()
    fused_ids: set = set()
    for group in optimizer.param_groups:
        ids = {id(p) for p in group["params"]}
        if group.get("use_muon"):
            muon_ids |= ids
            fused_ids |= set(group.get("_fused_gate_up_ids", set()))
        else:
            adamw_ids |= ids
    return muon_ids, adamw_ids, fused_ids


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


class FakeCudaTensor:
    def __init__(self, *, dtype: torch.dtype, shape: tuple[int, int]):
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self._shape = shape

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]


def test_find_best_restarts_matches_default_muon_coefficients():
    coefficients = expand_ns_coefficients((3.4445, -4.775, 2.0315), 5)

    assert find_best_restarts(coefficients, num_restarts=1) == [3]


def test_muon_gram_newton_schulz_updates_parameters_and_autotunes_restart():
    param = nn.Parameter(torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32))
    optimizer = Muon(
        [param],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
        gram_newton_schulz_num_restarts=1,
    )

    before = param.detach().clone()
    param.grad = torch.tensor([[0.5, -0.25, 0.75], [-1.0, 0.5, -0.5]], dtype=torch.float32)

    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.allclose(param, before)
    assert len(optimizer._gram_ns_orthogonalizers) == 1
    orthogonalizer = next(iter(optimizer._gram_ns_orthogonalizers.values()))
    assert list(orthogonalizer.reset_iterations) == [3]


def test_build_optimizer_threads_gram_newton_schulz_kwargs():
    model = TinyModule()

    optimizer = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="muon",
        no_decay_params=["bias"],
        optimizer_kwargs={
            "muon_lr": 0.02,
            "muon_ns_algorithm": "gram_newton_schulz",
            "muon_ns_use_quack_kernels": False,
            "muon_gram_ns_num_restarts": 1,
            "muon_grouped_gram_ns_fp32_byte_limit": 23,
            "muon_grad_dtype": "fp32",
            "muon_update_dtype": "bf16",
            "muon_force_momentum_path": True,
            "muon_fallback_optimizer": "sgd",
        },
    )

    assert isinstance(optimizer, Muon)
    muon_groups = [group for group in optimizer.param_groups if group["use_muon"]]
    adamw_groups = [group for group in optimizer.param_groups if not group["use_muon"]]

    assert muon_groups
    assert adamw_groups
    assert all(group["ns_algorithm"] == "gram_newton_schulz" for group in muon_groups)
    assert all(group["ns_use_quack_kernels"] is False for group in muon_groups)
    assert all(group["gram_newton_schulz_num_restarts"] == 1 for group in muon_groups)
    assert all(group["grouped_gram_ns_fp32_byte_limit"] == 23 for group in muon_groups)
    assert all(group["fallback_optimizer"] == "sgd" for group in adamw_groups)
    assert optimizer._momentum_dtype is torch.bfloat16
    assert optimizer._grad_dtype is torch.float32
    assert optimizer._update_dtype is torch.bfloat16
    assert optimizer._force_momentum_path is True
    assert optimizer._fallback_optimizer == "sgd"


def test_muon_sgd_fallback_updates_without_state():
    model = TinyModule()
    optimizer = build_optimizer(
        model,
        lr=0.1,
        optimizer_type="muon",
        optimizer_kwargs={
            "muon_momentum": 0.0,
            "muon_ns_use_quack_kernels": False,
            "muon_fallback_optimizer": "sgd",
        },
    )

    before_bias = model.linear.bias.detach().clone()
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    optimizer.step()

    assert not torch.allclose(model.linear.bias, before_bias)
    assert model.linear.bias not in optimizer.state


def test_muon_rejects_nonpositive_grouped_gram_newton_schulz_byte_limit():
    param = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))

    with pytest.raises(ValueError, match="grouped_gram_ns_fp32_byte_limit must be positive"):
        Muon([param], grouped_gram_ns_fp32_byte_limit=0)


def test_make_quack_backend_prefers_installed_quack(monkeypatch):
    gemm_calls = []
    fake_gemm_interface = SimpleNamespace(
        gemm=lambda A, B, tuned=True: gemm_calls.append(("gemm", tuned)) or ("gemm", A, B),
        gemm_add=lambda A, B, C=None, beta=1.0, tuned=True: (
            gemm_calls.append(("gemm_add", tuned)) or ("gemm_add", A, B, C, beta)
        ),
        gemm_symmetric=lambda A, B, C=None, alpha=1.0, beta=1.0: ("gemm_symmetric", A, B, C, alpha, beta),
    )
    import_calls = []

    def fake_import_module(module_name):
        import_calls.append(module_name)
        if module_name == "quack.gemm_interface":
            return fake_gemm_interface
        raise AssertionError(f"Unexpected import {module_name}")

    monkeypatch.setattr(gram_newton_schulz.importlib, "import_module", fake_import_module)
    gram_newton_schulz._make_quack_backend.cache_clear()

    backend = gram_newton_schulz._make_quack_backend()

    assert import_calls == ["quack.gemm_interface"]
    assert backend.sym_baddbmm("A", "B", "C", alpha=2.0, beta=3.0) == ("gemm_symmetric", "A", "B", "C", 2.0, 3.0)
    assert backend.mm("A", "B") == ("gemm", "A", "B")
    assert backend.mm_add("A", "B", "C", beta=3.0) == ("gemm_add", "A", "B", "C", 3.0)
    assert gemm_calls == [("gemm", False), ("gemm_add", False)]

    gram_newton_schulz._make_quack_backend.cache_clear()


def test_make_quack_backend_can_enable_tuned_gemm_for_muon(monkeypatch):
    gemm_calls = []
    fake_gemm_interface = SimpleNamespace(
        gemm=lambda A, B, tuned=True: gemm_calls.append(("gemm", tuned)) or ("gemm", A, B),
        gemm_add=lambda A, B, C=None, beta=1.0, tuned=True: (
            gemm_calls.append(("gemm_add", tuned)) or ("gemm_add", A, B, C, beta)
        ),
        gemm_symmetric=lambda A, B, C=None, alpha=1.0, beta=1.0: ("gemm_symmetric", A, B, C, alpha, beta),
    )

    monkeypatch.setenv("XORL_MUON_QUACK_TUNED", "1")
    monkeypatch.setattr(gram_newton_schulz.importlib, "import_module", lambda _: fake_gemm_interface)
    gram_newton_schulz._make_quack_backend.cache_clear()

    backend = gram_newton_schulz._make_quack_backend()

    assert backend.mm("A", "B") == ("gemm", "A", "B")
    assert backend.mm_add("A", "B", "C") == ("gemm_add", "A", "B", "C", 1.0)
    assert gemm_calls == [("gemm", True), ("gemm_add", True)]

    gram_newton_schulz._make_quack_backend.cache_clear()


def test_make_quack_backend_requires_installed_quack(monkeypatch):
    def fake_import_module(module_name):
        assert module_name == "quack.gemm_interface"
        raise ModuleNotFoundError("No module named 'quack'")

    monkeypatch.setattr(gram_newton_schulz.importlib, "import_module", fake_import_module)
    gram_newton_schulz._make_quack_backend.cache_clear()

    with pytest.raises(ImportError, match="upstream `quack-kernels` package"):
        gram_newton_schulz._make_quack_backend()

    gram_newton_schulz._make_quack_backend.cache_clear()


def test_select_backend_falls_back_to_torch_for_fp32_on_sm90(monkeypatch):
    orthogonalizer = GramNewtonSchulzOrthogonalizer(
        ns_coefficients=((3.4445, -4.775, 2.0315),) * 5,
        ns_use_quack_kernels=True,
    )
    quack_backend = object()

    monkeypatch.setattr(gram_newton_schulz, "_make_quack_backend", lambda: quack_backend)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))

    backend = orthogonalizer._select_backend(FakeCudaTensor(dtype=torch.float32, shape=(512, 512)))

    assert backend is gram_newton_schulz._TORCH_BACKEND


def test_select_backend_keeps_quack_for_bf16_on_sm90(monkeypatch):
    orthogonalizer = GramNewtonSchulzOrthogonalizer(
        ns_coefficients=((3.4445, -4.775, 2.0315),) * 5,
        ns_use_quack_kernels=True,
    )
    quack_backend = object()

    monkeypatch.setattr(gram_newton_schulz, "_make_quack_backend", lambda: quack_backend)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))

    backend = orthogonalizer._select_backend(FakeCudaTensor(dtype=torch.bfloat16, shape=(512, 512)))

    assert backend is quack_backend


def test_muon_groups_gram_newton_schulz_updates_by_shape(monkeypatch):
    class FakeOrthogonalizer:
        def __init__(self):
            self.seen_shapes = []

        def orthogonalize(self, X):
            self.seen_shapes.append(tuple(X.shape))
            if X.ndim == 3:
                offsets = torch.arange(1, X.shape[0] + 1, device=X.device, dtype=X.dtype).view(-1, 1, 1)
                return X + offsets
            return X + 7

    p1 = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    p2 = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    p3 = nn.Parameter(torch.zeros((2, 2), dtype=torch.float32))
    optimizer = Muon(
        [p1, p2, p3],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(optimizer, "_get_gram_ns_orthogonalizer", lambda group: orthogonalizer)

    p1.grad = torch.ones((2, 3), dtype=torch.float32)
    p2.grad = torch.full((2, 3), 2.0, dtype=torch.float32)
    p3.grad = torch.full((2, 2), 3.0, dtype=torch.float32)

    optimizer.step()

    assert orthogonalizer.seen_shapes == [(2, 2, 3), (2, 2)]
    assert torch.allclose(p1, torch.full((2, 3), -2.0))
    assert torch.allclose(p2, torch.full((2, 3), -4.0))
    assert torch.allclose(p3, torch.full((2, 2), -10.0))


def test_muon_groups_gram_newton_schulz_updates_by_matrix_shape(monkeypatch):
    class FakeOrthogonalizer:
        def __init__(self):
            self.seen_shapes = []

        def orthogonalize(self, X):
            self.seen_shapes.append(tuple(X.shape))
            if X.ndim == 3:
                offsets = torch.arange(1, X.shape[0] + 1, device=X.device, dtype=X.dtype).view(-1, 1, 1)
                return X + offsets
            return X + 7

    p1 = nn.Parameter(torch.zeros((2, 3, 4), dtype=torch.float32))
    p2 = nn.Parameter(torch.zeros((1, 3, 4), dtype=torch.float32))
    p3 = nn.Parameter(torch.zeros((3, 5), dtype=torch.float32))
    optimizer = Muon(
        [p1, p2, p3],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(optimizer, "_get_gram_ns_orthogonalizer", lambda group: orthogonalizer)

    p1.grad = torch.ones((2, 3, 4), dtype=torch.float32)
    p2.grad = torch.full((1, 3, 4), 2.0, dtype=torch.float32)
    p3.grad = torch.full((3, 5), 3.0, dtype=torch.float32)

    optimizer.step()

    assert orthogonalizer.seen_shapes == [(3, 3, 4), (3, 5)]
    expected_p1 = torch.tensor(
        [
            [[-2.0, -2.0, -2.0, -2.0], [-2.0, -2.0, -2.0, -2.0], [-2.0, -2.0, -2.0, -2.0]],
            [[-3.0, -3.0, -3.0, -3.0], [-3.0, -3.0, -3.0, -3.0], [-3.0, -3.0, -3.0, -3.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(p1, expected_p1)
    assert torch.allclose(p2, torch.full((1, 3, 4), -5.0))
    assert torch.allclose(p3, torch.full((3, 5), -10.0))


def test_muon_groups_gram_newton_schulz_transpose_equivalent_shapes(monkeypatch):
    class FakeOrthogonalizer:
        def __init__(self):
            self.seen_shapes = []

        def orthogonalize(self, X):
            self.seen_shapes.append(tuple(X.shape))
            offsets = torch.arange(1, X.shape[0] + 1, device=X.device, dtype=X.dtype).view(-1, 1, 1)
            return X + offsets

    p1 = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    p2 = nn.Parameter(torch.zeros((3, 2), dtype=torch.float32))
    optimizer = Muon(
        [p1, p2],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(optimizer, "_get_gram_ns_orthogonalizer", lambda group: orthogonalizer)

    p1.grad = torch.ones((2, 3), dtype=torch.float32)
    p2.grad = torch.full((3, 2), 2.0, dtype=torch.float32)

    optimizer.step()

    assert orthogonalizer.seen_shapes == [(2, 2, 3)]
    assert torch.allclose(p1, torch.full((2, 3), -2.0))
    assert torch.allclose(p2, torch.full((3, 2), -4.0))


def test_muon_groups_fused_gate_up_halves(monkeypatch):
    class FakeOrthogonalizer:
        def __init__(self):
            self.seen_shapes = []

        def orthogonalize(self, X):
            self.seen_shapes.append(tuple(X.shape))
            offsets = torch.arange(1, X.shape[0] + 1, device=X.device, dtype=X.dtype).view(-1, *([1] * (X.ndim - 1)))
            return X + offsets

    p1 = nn.Parameter(torch.zeros((2, 3, 4), dtype=torch.float32))
    p2 = nn.Parameter(torch.zeros((2, 3, 4), dtype=torch.float32))
    optimizer = Muon(
        [
            {
                "params": [p1, p2],
                "_fused_gate_up_ids": {id(p1), id(p2)},
            }
        ],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(optimizer, "_get_gram_ns_orthogonalizer", lambda group: orthogonalizer)

    p1.grad = torch.ones((2, 3, 4), dtype=torch.float32)
    p2.grad = torch.ones((2, 3, 4), dtype=torch.float32)

    optimizer.step()

    assert orthogonalizer.seen_shapes == [(8, 2, 3)]
    expected_p1 = torch.tensor(
        [
            [[-2.0, -2.0, -4.0, -4.0], [-2.0, -2.0, -4.0, -4.0], [-2.0, -2.0, -4.0, -4.0]],
            [[-3.0, -3.0, -5.0, -5.0], [-3.0, -3.0, -5.0, -5.0], [-3.0, -3.0, -5.0, -5.0]],
        ],
        dtype=torch.float32,
    )
    expected_p2 = torch.tensor(
        [
            [[-6.0, -6.0, -8.0, -8.0], [-6.0, -6.0, -8.0, -8.0], [-6.0, -6.0, -8.0, -8.0]],
            [[-7.0, -7.0, -9.0, -9.0], [-7.0, -7.0, -9.0, -9.0], [-7.0, -7.0, -9.0, -9.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(p1, expected_p1)
    assert torch.allclose(p2, expected_p2)


def test_muon_standard_newton_schulz_preserves_batched_leading_dims(monkeypatch):
    seen_shapes = []

    def fake_batched_zeropower(update, ns_coefficients, ns_steps, eps):
        # Receives the flattened-batch tensor [B, H, I]; assign each batch element
        # a distinct constant offset so we can confirm correct un-flattening.
        seen_shapes.append(tuple(update.shape))
        offsets = torch.arange(1, update.shape[0] + 1, dtype=update.dtype, device=update.device).reshape(-1, 1, 1)
        return update + offsets

    p = nn.Parameter(torch.zeros((2, 3, 4), dtype=torch.float32))
    optimizer = Muon(
        [p],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="standard_newton_schulz",
    )

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(muon_module, "_batched_zeropower_via_newtonschulz", fake_batched_zeropower)

    p.grad = torch.ones((2, 3, 4), dtype=torch.float32)

    optimizer.step()

    # Single batched call over the flattened leading dims: [B=2, H=3, I=4].
    assert seen_shapes == [(2, 3, 4)]
    expected = torch.tensor(
        [
            [[-2.0, -2.0, -2.0, -2.0], [-2.0, -2.0, -2.0, -2.0], [-2.0, -2.0, -2.0, -2.0]],
            [[-3.0, -3.0, -3.0, -3.0], [-3.0, -3.0, -3.0, -3.0], [-3.0, -3.0, -3.0, -3.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(p, expected)


def test_muon_chunks_grouped_gram_newton_schulz_batches(monkeypatch):
    class FakeOrthogonalizer:
        def __init__(self):
            self.seen_shapes = []

        def orthogonalize(self, X):
            self.seen_shapes.append(tuple(X.shape))
            return X + 1

    p1 = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    p2 = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    optimizer = Muon(
        [p1, p2],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="gram_newton_schulz",
        ns_use_quack_kernels=False,
        grouped_gram_ns_fp32_byte_limit=23,
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(optimizer, "_get_gram_ns_orthogonalizer", lambda group: orthogonalizer)

    p1.grad = torch.ones((2, 3), dtype=torch.float32)
    p2.grad = torch.ones((2, 3), dtype=torch.float32)

    optimizer.step()

    assert orthogonalizer.seen_shapes == [(2, 3), (2, 3)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for dtype-preservation coverage")
def test_gram_newton_schulz_preserves_fp32_compute_dtype_on_cuda():
    seen_dtypes = []

    def sym_mm(A, B):
        seen_dtypes.append(A.dtype)
        return A @ B

    def sym_baddbmm(A, B, C, alpha=1.0, beta=1.0):
        seen_dtypes.extend((A.dtype, B.dtype, C.dtype))
        if A.ndim == 2:
            return torch.addmm(C, A, B, beta=beta, alpha=alpha)
        return torch.baddbmm(C, A, B, beta=beta, alpha=alpha)

    def mm(A, B):
        seen_dtypes.append(A.dtype)
        return A @ B

    def mm_add(A, B, C, beta=1.0):
        seen_dtypes.extend((A.dtype, B.dtype, C.dtype))
        if A.ndim == 2:
            return torch.addmm(C, A, B, beta=beta)
        return torch.baddbmm(C, A, B, beta=beta)

    orthogonalizer = GramNewtonSchulzOrthogonalizer(
        ns_coefficients=((3.4445, -4.775, 2.0315),) * 5,
        ns_use_quack_kernels=False,
    )
    orthogonalizer._select_backend = lambda X: SimpleNamespace(
        sym_mm=sym_mm,
        sym_baddbmm=sym_baddbmm,
        mm=mm,
        mm_add=mm_add,
    )

    input_matrix = torch.randn(8, 8, device="cuda", dtype=torch.float32)
    output_matrix = orthogonalizer.orthogonalize(input_matrix)

    assert output_matrix.dtype is torch.float32
    assert seen_dtypes
    assert all(dtype is torch.float32 for dtype in seen_dtypes)


def test_collect_fused_gate_up_ids_gated_vs_non_gated_experts():
    gated = MoEExperts(num_experts=2, hidden_dim=4, intermediate_size=3, moe_implementation="eager")
    assert getattr(gated.gate_up_proj, "_fused_gate_up", False)
    assert id(gated.gate_up_proj) in _collect_fused_gate_up_ids(gated)

    non_gated = MoEExperts(
        num_experts=2,
        hidden_dim=4,
        intermediate_size=3,
        hidden_act="relu2",
        moe_implementation="eager",
        gated=False,
    )
    assert not getattr(non_gated.gate_up_proj, "_fused_gate_up", False)
    assert id(non_gated.gate_up_proj) not in _collect_fused_gate_up_ids(non_gated)


def test_collect_fused_gate_up_ids_survives_parameter_attribute_loss():
    """FSDP2's fully_shard swaps params for fresh DTensor params, dropping
    python attributes; detection must still work via the owning module."""
    gated = MoEExperts(num_experts=2, hidden_dim=4, intermediate_size=3, moe_implementation="eager")
    gated.gate_up_proj = nn.Parameter(gated.gate_up_proj.detach().clone())
    assert not getattr(gated.gate_up_proj, "_fused_gate_up", False)

    assert id(gated.gate_up_proj) in _collect_fused_gate_up_ids(gated)


def test_muon_fused_split_set_gated_deepseek_v3_vs_non_gated_nemotron_h():
    deepseek = DeepseekV3ForCausalLM(_deepseek_v3_tiny_config())
    _, _, deepseek_fused_ids = _split_group_ids(_build_trainer_style_muon(deepseek))
    deepseek_gate_up = [(n, p) for n, p in deepseek.named_parameters() if "gate_up_proj" in n]
    assert deepseek_gate_up
    for name, param in deepseek_gate_up:
        assert id(param) in deepseek_fused_ids, name

    nemotron = _build_nemotron_h_model()
    nemotron_muon_ids, _, nemotron_fused_ids = _split_group_ids(_build_trainer_style_muon(nemotron))
    nemotron_gate_up = [(n, p) for n, p in nemotron.named_parameters() if "gate_up_proj" in n]
    assert nemotron_gate_up
    for name, param in nemotron_gate_up:
        # Non-gated experts: Muon-eligible, but never split in half for NS.
        assert id(param) in nemotron_muon_ids, name
        assert id(param) not in nemotron_fused_ids, name


def test_muon_classification_nemotron_h():
    model = _build_nemotron_h_model()
    muon_ids, adamw_ids, _ = _split_group_ids(_build_trainer_style_muon(model))
    params = dict(model.named_parameters())
    assert {id(p) for p in params.values()} == muon_ids | adamw_ids

    for name, param in params.items():
        # 1D params (A_log, D, dt_bias, conv1d.bias, norms) and the depthwise
        # conv1d.weight [conv_dim, 1, k] must not get Newton-Schulz.
        if param.ndim < 2 or "conv1d" in name:
            assert id(param) in adamw_ids, name

    # Embeddings, lm_head, and the router gate are excluded from Muon.
    assert id(params["model.embeddings.weight"]) in adamw_ids
    assert id(params["lm_head.weight"]) in adamw_ids
    assert id(params["model.layers.2.mixer.gate.weight"]) in adamw_ids

    # Dense projections (mamba in/out, attention q/k/v/o, MoE latent and
    # shared-expert projections) and 3D expert weights go to Muon.
    muon_patterns = (
        "mixer.in_proj.weight",
        "mixer.out_proj.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "fc1_latent_proj.weight",
        "fc2_latent_proj.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    )
    for pattern in muon_patterns:
        matching = [n for n in params if pattern in n]
        assert matching, pattern
        for name in matching:
            assert id(params[name]) in muon_ids, name


def test_muon_step_on_tiny_nemotron_h_updates_params():
    model = _build_nemotron_h_model()
    model.train()
    optimizer = _build_trainer_style_muon(model)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    outputs = model(input_ids=input_ids)
    outputs.last_hidden_state.float().pow(2).mean().backward()

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer.step()

    changed = {name for name, param in model.named_parameters() if not torch.equal(param, before[name])}
    for expected in (
        "model.layers.2.mixer.experts.gate_up_proj",  # non-gated 3D expert weight (Muon, unsplit)
        "model.layers.0.mixer.in_proj.weight",  # mamba dense (Muon)
        "model.layers.0.mixer.conv1d.weight",  # depthwise conv (AdamW fallback)
        "model.layers.0.mixer.A_log",  # 1D (AdamW fallback)
        "model.layers.1.mixer.q_proj.weight",  # attention dense (Muon)
    ):
        assert expected in changed, expected
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name
