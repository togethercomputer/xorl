"""Model-level parity: shared-prefix repack + forward + loss == standard path.

Runs a small real Qwen3 over a packed micro-batch with shared-prefix groups,
both standard and via repack -> forward (shared_prefix_context flows to
attention) -> loss on the repacked layout, and asserts the policy loss matches
and the remapped per-token logprobs match at trained positions.
"""

import pytest
import torch


pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3Model  # noqa: E402
from xorl.ops.loss.policy_loss import policy_loss_function  # noqa: E402
from xorl.ops.shared_prefix import shared_prefix_remap_to_original, shared_prefix_repack_batch  # noqa: E402


IGNORE = -100


def _build_packed(seqs, device):
    ids, labels, pos, adv, old_lp, cu = [], [], [], [], [], [0]
    for prompt, resp in seqs:
        seq = list(prompt) + list(resp)
        p = len(prompt)
        lab = [IGNORE] * len(seq)
        for j in range(p - 1, len(seq) - 1):
            lab[j] = seq[j + 1]
        ids += seq
        labels += lab
        pos += list(range(len(seq)))
        adv += [0.3 * (j % 5) for j in range(len(seq))]
        old_lp += [-0.1 * (j % 7) for j in range(len(seq))]
        cu.append(len(ids))
    t = lambda x, dt: torch.tensor(x, device=device, dtype=dt)  # noqa: E731
    return {
        "input_ids": t(ids, torch.long).unsqueeze(0),
        "target_tokens": t(labels, torch.long).unsqueeze(0),
        "position_ids": t(pos, torch.long).unsqueeze(0),
        "advantages": t(adv, torch.float).unsqueeze(0),
        "logprobs": t(old_lp, torch.float).unsqueeze(0),
        "cu_seq_lens_q": t(cu, torch.int32),
    }


def _make_model(device):
    config = Qwen3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "flash_attention_3"
    torch.manual_seed(0)
    return Qwen3Model(config).to(device=device, dtype=torch.bfloat16).eval()


def test_model_policy_loss_and_logprob_parity():
    device = "cuda"
    model = _make_model(device)
    seqs = [
        ([5, 6, 7, 8], [101, 102, 103]),
        ([5, 6, 7, 8], [201, 202]),
        ([5, 6, 7, 8], [203, 204, 205, 206]),
        ([11, 12], [301, 302, 303]),
        ([11, 12], [304]),
        ([9, 3, 1], [401, 402]),  # singleton
    ]
    b = _build_packed(seqs, device)
    T = b["input_ids"].size(1)
    cu = b["cu_seq_lens_q"]
    max_len = int((cu[1:] - cu[:-1]).max().item())
    weight = model.embed_tokens.weight

    def loss_of(hidden, labels, adv, old_lp):
        return policy_loss_function(
            hidden_states=hidden.float(),
            weight=weight.float(),
            labels=labels,
            old_logprobs=old_lp,
            advantages=adv,
            ce_mode="eager",
        )

    # --- standard ---
    std_hidden = model(
        input_ids=b["input_ids"],
        position_ids=b["position_ids"],
        attention_mask=torch.ones((1, T), dtype=torch.long, device=device),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max_len,
        max_length_k=max_len,
    ).last_hidden_state
    std = loss_of(std_hidden, b["target_tokens"], b["advantages"], b["logprobs"])

    # --- shared-prefix (repack -> forward -> loss on repacked layout) ---
    rp = shared_prefix_repack_batch(b)
    assert rp is not None
    ctx = rp["shared_prefix_context"]
    sp_hidden = model(
        input_ids=rp["input_ids"],
        position_ids=rp["position_ids"],
        attention_mask=rp["attention_mask"],
        shared_prefix_context=ctx,
    ).last_hidden_state
    sp = loss_of(sp_hidden, rp["target_tokens"], rp["advantages"], rp["logprobs"])

    # loss scalar parity (same valid-token set, same values)
    torch.testing.assert_close(sp.loss, std.loss, atol=2e-2, rtol=2e-2)

    # remap per-token logprobs back to original; compare at trained positions
    sp_lp_orig = shared_prefix_remap_to_original(sp.per_token_logprobs, ctx)
    valid = (b["target_tokens"].squeeze(0) != IGNORE).nonzero(as_tuple=True)[0]
    torch.testing.assert_close(
        sp_lp_orig.squeeze(0)[valid], std.per_token_logprobs.squeeze(0)[valid], atol=2e-2, rtol=2e-2
    )
