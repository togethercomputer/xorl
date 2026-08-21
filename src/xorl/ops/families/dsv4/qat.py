import torch


def fp8_simulate(x: torch.Tensor, block_size: int):
    # Lazy import: ``act_quant`` requires tilelang. Importing this module on a
    # tilelang-less host stays valid as long as ``fp8_simulate_qat`` is never
    # actually called (the QAT path is gated by the model's FP8 config).
    from .kernel.act_quant import act_quant  # noqa: PLC0415

    y, scale = act_quant(x.contiguous(), block_size, "ue8m0")
    y = y.unflatten(-1, (-1, block_size)).float() * scale.unsqueeze(-1)
    return y.flatten(-2).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
