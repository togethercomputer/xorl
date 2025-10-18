import triton


def innermost_fn(fn: triton.KernelInterface):
    while hasattr(fn, "fn"):
        fn = fn.fn
    return fn


def qualified_name(fn: triton.KernelInterface) -> str:
    return innermost_fn(fn).__qualname__
