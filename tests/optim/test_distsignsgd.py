from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xorl.optim.distsignsgd as distsign_module
import xorl.optim.optimizer as optimizer_module
from xorl.optim import DistSignSGD, build_optimizer


pytestmark = [pytest.mark.cpu]


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


class LocalOnlyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))


class RecordingReduceScatter:
    def __init__(self):
        self.seen_input = None
        self.seen_op = None
        self.seen_async_op = None

    def allocate(self, size, *, dtype, device):
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        self.seen_input = input_tensor.clone()
        self.seen_op = op
        self.seen_async_op = async_op
        output_tensor.copy_(input_tensor[: output_tensor.numel()])
        return None


def test_distsignsgd_applies_preaggregated_gradient():
    param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = DistSignSGD([param], lr=0.1)

    param.grad = torch.tensor([0.5, -0.25, 0.0])
    optimizer.step()

    expected = torch.tensor([0.95, -1.975, 3.0])
    assert torch.allclose(param, expected)


def test_distsignsgd_applies_decoupled_weight_decay_before_preaggregated_update():
    param = nn.Parameter(torch.tensor([2.0, -2.0]))
    optimizer = DistSignSGD([param], lr=0.1, weight_decay=0.5)

    param.grad = torch.tensor([0.25, -0.5])
    optimizer.step()

    expected = torch.tensor([1.875, -1.85])
    assert torch.allclose(param, expected)


def test_distsignsgd_rejects_sparse_gradients():
    param = nn.Parameter(torch.ones(4))
    optimizer = DistSignSGD([param], lr=0.1)
    param.grad = torch.sparse_coo_tensor(indices=[[0, 2]], values=torch.tensor([1.0, -1.0]), size=(4,))

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optimizer.step()


def test_distsignsgd_keeps_optimizer_state_empty_across_steps():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = DistSignSGD([param], lr=0.1)

    for grad in (torch.tensor([1.0]), torch.tensor([-0.5])):
        param.grad = grad
        optimizer.step()
        assert len(optimizer.state) == 0


def test_distsignsgd_state_dict_round_trips_without_state_tensors():
    source_param = nn.Parameter(torch.tensor([1.0, -1.0]))
    source_optimizer = DistSignSGD([source_param], lr=0.1, weight_decay=0.25)
    source_param.grad = torch.tensor([0.5, -0.25])
    source_optimizer.step()

    state_dict = source_optimizer.state_dict()

    target_param = nn.Parameter(torch.tensor([0.0, 0.0]))
    target_optimizer = DistSignSGD([target_param], lr=1.0, weight_decay=0.0)
    target_optimizer.load_state_dict(state_dict)

    assert state_dict["state"] == {}
    assert target_optimizer.state_dict()["state"] == {}
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert target_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.25)


def test_dist_sign_reduce_scatter_signs_input_before_reduce():
    inner = RecordingReduceScatter()
    comm = distsign_module.DistSignReduceScatter(inner_comm=inner)

    input_tensor = torch.tensor([2.0, -0.5, 0.0], dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    comm(
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        group=None,
        op=torch.distributed.ReduceOp.SUM,
        async_op=True,
    )

    expected = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    assert torch.equal(input_tensor, expected)
    assert torch.equal(inner.seen_input, expected)
    assert torch.equal(output_tensor, expected)
    assert inner.seen_op == torch.distributed.ReduceOp.SUM
    assert inner.seen_async_op is True


def test_dist_sign_reduce_scatter_forces_sum_when_caller_passes_avg():
    inner = RecordingReduceScatter()
    comm = distsign_module.DistSignReduceScatter(inner_comm=inner)

    input_tensor = torch.tensor([2.0, -0.5, 0.0], dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    comm(
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        group=None,
        op=torch.distributed.ReduceOp.AVG,
    )

    # FSDP2 with reduce_dtype=fp32 may pass AVG; the trainer's voter-total
    # divisor would then double-divide. Forcing SUM here keeps the sign-vote
    # accumulator semantics intact.
    assert inner.seen_op == torch.distributed.ReduceOp.SUM


def test_dist_sign_reduce_scatter_sums_sp_before_sign(monkeypatch):
    inner = RecordingReduceScatter()
    comm = distsign_module.DistSignReduceScatter(inner_comm=inner, sp_group="sp-group")
    reduced = []

    def fake_all_reduce(tensor, op, group):
        reduced.append((tensor.clone(), op, group))
        tensor.add_(torch.tensor([-3.0, 1.0, 0.0], dtype=tensor.dtype))

    monkeypatch.setattr(distsign_module.dist, "all_reduce", fake_all_reduce)

    input_tensor = torch.tensor([2.0, -0.5, 0.0], dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    comm(
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        group=None,
        op=torch.distributed.ReduceOp.SUM,
    )

    expected = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32)
    assert len(reduced) == 1
    assert reduced[0][1] == torch.distributed.ReduceOp.SUM
    assert reduced[0][2] == "sp-group"
    assert torch.equal(input_tensor, expected)
    assert torch.equal(inner.seen_input, expected)
    assert torch.equal(output_tensor, expected)


def test_configure_distsignsgd_registers_local_sign_hook(monkeypatch):
    model = LocalOnlyModule()

    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2",
            dp_replicate_enabled=False,
            ep_enabled=False,
            cp_enabled=False,
            cp_fsdp_mode="none",
            sp_grad_sync_group=None,
        ),
    )

    distsign_module.configure_distsignsgd(model)

    (model.param * torch.tensor([2.0, -3.0])).sum().backward()
    (model.param * torch.tensor([-4.0, 5.0])).sum().backward()

    assert torch.equal(model.param.grad, torch.tensor([0.0, 0.0]))
    assert getattr(model.param, "_distsign_local_hook_registered", False) is True
    assert getattr(model, "_distsignsgd_configured", False) is True


def test_configure_distsignsgd_skips_local_sign_hook_for_fsdp_managed_params(monkeypatch):
    class FakeFSDPModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.managed = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))

        def _get_fsdp_state(self):
            return SimpleNamespace(
                _fsdp_param_group=SimpleNamespace(
                    fsdp_params=[SimpleNamespace(sharded_param=self.managed)],
                )
            )

        def set_custom_reduce_scatter(self, comm):
            self._reduce_scatter_comm = comm

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fsdp = FakeFSDPModule()
            self.local = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))

    model = MixedModel()

    monkeypatch.setattr(distsign_module, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2",
            dp_replicate_enabled=False,
            ep_enabled=False,
            cp_enabled=False,
            cp_fsdp_mode="none",
            sp_grad_sync_group=None,
        ),
    )

    distsign_module.configure_distsignsgd(model)

    (model.fsdp.managed * torch.tensor([2.0, -3.0])).sum().backward()
    (model.fsdp.managed * torch.tensor([-4.0, 5.0])).sum().backward()
    (model.local * torch.tensor([2.0, -3.0])).sum().backward()
    (model.local * torch.tensor([-4.0, 5.0])).sum().backward()

    assert torch.equal(model.fsdp.managed.grad, torch.tensor([-2.0, 2.0]))
    assert torch.equal(model.local.grad, torch.tensor([0.0, 0.0]))
    assert getattr(model.local, "_distsign_local_hook_registered", False) is True
    assert getattr(model.fsdp.managed, "_distsign_local_hook_registered", False) is False


def test_configure_distsignsgd_rejects_hsdp(monkeypatch):
    model = LocalOnlyModule()

    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_mode="fsdp2", dp_replicate_enabled=True),
    )

    with pytest.raises(NotImplementedError, match="does not yet support HSDP"):
        distsign_module.configure_distsignsgd(model)


def test_configure_distsignsgd_rejects_sequence_parallel_folded_into_fsdp(monkeypatch):
    model = LocalOnlyModule()

    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2",
            dp_replicate_enabled=False,
            ep_enabled=False,
            cp_enabled=True,
            cp_fsdp_mode="all",
        ),
    )

    with pytest.raises(NotImplementedError, match="set cp_fsdp_mode='none'"):
        distsign_module.configure_distsignsgd(model)


def test_configure_distsignsgd_rejects_expert_parallelism(monkeypatch):
    model = LocalOnlyModule()

    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2",
            dp_replicate_enabled=False,
            ep_enabled=True,
            cp_enabled=False,
            cp_fsdp_mode="none",
        ),
    )

    with pytest.raises(NotImplementedError, match="expert parallelism"):
        distsign_module.configure_distsignsgd(model)


def test_configure_distsignsgd_rejects_non_fsdp_dtensor_param(monkeypatch):
    class FakeDTensor:
        pass

    class TPOnlyModule(nn.Module):
        def __init__(self):
            super().__init__()
            tp_param = nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
            # Spoof a non-FSDP DTensor parameter.
            tp_param.__class__ = type("DTensorParam", (FakeDTensor, nn.Parameter), {})
            self.tp_param = tp_param

    model = TPOnlyModule()

    monkeypatch.setattr(distsign_module, "DTensor", FakeDTensor)
    monkeypatch.setattr(
        distsign_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            dp_mode="fsdp2",
            dp_replicate_enabled=False,
            ep_enabled=False,
            cp_enabled=False,
            cp_fsdp_mode="none",
            sp_grad_sync_group=None,
        ),
    )

    with pytest.raises(NotImplementedError, match="DTensor parameter that is not FSDP-managed"):
        distsign_module.configure_distsignsgd(model)


def test_build_optimizer_supports_distsignsgd_and_preserves_weight_decay_split(monkeypatch):
    model = TinyModule()
    configured = []

    monkeypatch.setattr(
        optimizer_module,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_mode="fsdp2", ep_enabled=False),
    )
    monkeypatch.setattr(
        optimizer_module, "configure_distsignsgd", lambda configured_model: configured.append(configured_model)
    )

    optimizer = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="distsignsgd",
        no_decay_params=["bias"],
    )

    assert isinstance(optimizer, DistSignSGD)
    assert configured == [model]
    assert len(optimizer.param_groups) == 2

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == pytest.approx(0.01)
    assert no_decay_group["weight_decay"] == pytest.approx(0.0)
    assert decay_group["params"] == [model.linear.weight]
    assert no_decay_group["params"] == [model.linear.bias]


def test_build_optimizer_rejects_distsignsgd_without_fsdp2(monkeypatch):
    model = TinyModule()

    monkeypatch.setattr(
        optimizer_module,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_mode="ddp"),
    )

    with pytest.raises(ValueError, match="requires data_parallel_mode='fsdp2'"):
        build_optimizer(model, optimizer_type="distsignsgd")
