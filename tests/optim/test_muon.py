from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xorl.optim.gram_newton_schulz as gram_newton_schulz
import xorl.optim.muon as muon_module
from xorl.optim import Muon, build_optimizer
from xorl.optim.gram_newton_schulz import GramNewtonSchulzOrthogonalizer, expand_ns_coefficients, find_best_restarts


pytestmark = [pytest.mark.cpu]


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
            "muon_grad_dtype": "fp32",
            "muon_update_dtype": "bf16",
            "muon_force_momentum_path": True,
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
    assert optimizer._momentum_dtype is torch.bfloat16
    assert optimizer._grad_dtype is torch.float32
    assert optimizer._update_dtype is torch.bfloat16
    assert optimizer._force_momentum_path is True


def test_make_quack_backend_prefers_installed_quack(monkeypatch):
    fake_gemm_interface = SimpleNamespace(
        gemm=lambda A, B: ("gemm", A, B),
        gemm_add=lambda A, B, C=None, beta=1.0: ("gemm_add", A, B, C, beta),
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
    call_count = 0

    def fake_zeropower(update, ns_coefficients, ns_steps, eps):
        nonlocal call_count
        call_count += 1
        seen_shapes.append(tuple(update.shape))
        return update + call_count

    p = nn.Parameter(torch.zeros((2, 3, 4), dtype=torch.float32))
    optimizer = Muon(
        [p],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_algorithm="standard_newton_schulz",
    )

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(muon_module, "_zeropower_via_newtonschulz", fake_zeropower)

    p.grad = torch.ones((2, 3, 4), dtype=torch.float32)

    optimizer.step()

    assert seen_shapes == [(3, 4), (3, 4)]
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
    )
    orthogonalizer = FakeOrthogonalizer()

    monkeypatch.setattr(muon_module, "_adjust_lr", lambda lr, adjust_lr_fn, shape: lr)
    monkeypatch.setattr(muon_module, "GROUPED_GRAM_NS_FP32_BYTE_LIMIT", 23)
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
