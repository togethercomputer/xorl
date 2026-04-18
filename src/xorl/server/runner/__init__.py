"""Lazy exports for server runner modules."""

__all__ = ["RunnerDispatcher", "ModelRunner"]


def __getattr__(name):
    if name == "ModelRunner":
        from xorl.server.runner.model_runner import ModelRunner  # noqa: PLC0415

        globals()["ModelRunner"] = ModelRunner
        return ModelRunner
    if name == "RunnerDispatcher":
        from xorl.server.runner.runner_dispatcher import RunnerDispatcher  # noqa: PLC0415

        globals()["RunnerDispatcher"] = RunnerDispatcher
        return RunnerDispatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
