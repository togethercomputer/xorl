from types import MethodType

import pytest

from xorl.models.transformers.glm5.index_share import IndexShareMode
from xorl.server.runner.model_runner import ModelRunner
from xorl.trainers.trainer import Trainer


class _RetainingModel:
    def __init__(self) -> None:
        self.release_count = 0

    def release_index_share_context(self) -> None:
        self.release_count += 1


@pytest.mark.cpu
def test_offline_forward_backward_failure_releases_retained_context():
    trainer = Trainer.__new__(Trainer)
    model = _RetainingModel()
    trainer.model = model
    trainer._all_model_parts = MethodType(lambda _self: [model], trainer)
    trainer._forward_backward_impl = MethodType(
        lambda _self, _micro_batches, _global_valid_tokens: (_ for _ in ()).throw(RuntimeError("loss failed")),
        trainer,
    )

    assert trainer._index_share_forward_kwargs(model, IndexShareMode.TRAINING_WITH_BACKWARD) == {
        "index_share_mode": IndexShareMode.TRAINING_WITH_BACKWARD
    }
    with pytest.raises(RuntimeError, match="loss failed"):
        trainer._forward_backward([], None)
    assert model.release_count == 1


@pytest.mark.cpu
def test_server_forward_loop_failure_releases_retained_context():
    runner = ModelRunner.__new__(ModelRunner)
    model = _RetainingModel()
    runner.model = model
    runner.model_parts = []
    runner._forward_loop_impl = MethodType(
        lambda _self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backward failed")),
        runner,
    )

    assert runner._index_share_forward_kwargs(IndexShareMode.FORWARD_ONLY) == {
        "index_share_mode": IndexShareMode.FORWARD_ONLY
    }
    with pytest.raises(RuntimeError, match="backward failed"):
        runner._forward_loop([], "causallm_loss", {}, compute_backward=True)
    assert model.release_count == 1
