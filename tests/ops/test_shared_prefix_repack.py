"""CPU tests for shared-prefix group detection, repack, and output remap."""

import pytest
import torch

from xorl.ops.shared_prefix import (
    SharedPrefixContext,
    detect_prompt_groups,
    shared_prefix_remap_to_original,
    shared_prefix_repack_batch,
)


pytestmark = pytest.mark.cpu

IGNORE = -100


def _build_packed(seqs):
    """Build a packed batch dict from ``seqs`` (each ``(prompt, response)``).

    Labels are next-token with the prompt interior + final position ignored, so
    the first valid label sits at ``P-1`` (it predicts response[0]).
    """
    ids, labels, pos, adv, cu = [], [], [], [], [0]
    for prompt, resp in seqs:
        seq = list(prompt) + list(resp)
        p = len(prompt)
        lab = [IGNORE] * len(seq)
        for j in range(p - 1, len(seq) - 1):
            lab[j] = seq[j + 1]
        ids += seq
        labels += lab
        pos += list(range(len(seq)))
        adv += [float(t) for t in range(len(seq))]  # distinct per-position marker
        cu.append(len(ids))
    return {
        "input_ids": torch.tensor(ids).unsqueeze(0),
        "target_tokens": torch.tensor(labels).unsqueeze(0),
        "position_ids": torch.tensor(pos).unsqueeze(0),
        "advantages": torch.tensor(adv).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor(cu, dtype=torch.int32),
    }


def test_detect_groups_shared_and_singleton():
    seqs = [
        ([10, 11, 12], [20, 21]),
        ([10, 11, 12], [22, 23, 24]),
        ([10, 11, 12], [25]),
        ([30, 31], [40, 41, 42]),
    ]
    b = _build_packed(seqs)
    groups, prompt_lens = detect_prompt_groups(b["input_ids"], b["target_tokens"], b["cu_seq_lens_q"])
    assert groups == [[0, 1, 2], [3]]
    assert prompt_lens == [3, 3, 3, 2]


def test_no_shared_prefix_returns_none():
    b = _build_packed([([1, 2], [3, 4]), ([5, 6], [7, 8])])
    assert shared_prefix_repack_batch(b) is None


def test_repack_layout_and_compression():
    seqs = [
        ([10, 11, 12], [20, 21]),
        ([10, 11, 12], [22, 23, 24]),
        ([10, 11, 12], [25]),
        ([30, 31], [40, 41, 42]),
    ]
    b = _build_packed(seqs)
    out = shared_prefix_repack_batch(b)
    assert out is not None
    ctx: SharedPrefixContext = out["shared_prefix_context"]

    # group A: shared prefix [10,11] (P-1=2), decoded blocks [12,20,21] [12,22,23,24] [12,25]
    # group B: shared prefix [30] (P-1=1),  decoded block [31,40,41,42]
    expected = [10, 11, 12, 20, 21, 12, 22, 23, 24, 12, 25, 30, 31, 40, 41, 42]
    assert out["input_ids"].squeeze(0).tolist() == expected
    assert ctx.repacked_len == len(expected) == 16
    assert ctx.orig_len == 20

    # boundary token p_{P-1} (12 / 31) duplicated into each decoded block, with its
    # original position (P-1) preserved for RoPE.
    assert out["position_ids"].squeeze(0).tolist() == [0, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 0, 1, 2, 3, 4]

    assert ctx.shared_idx.tolist() == [0, 1, 11]  # [10,11] (groupA) + [30] (groupB)
    assert ctx.cu_shared.tolist() == [0, 2, 3]
    assert ctx.cu_dec.tolist() == [0, 3, 7, 9, 13]  # block lens 3,4,2,4
    assert ctx.cu_cross_q.tolist() == [0, 9, 13]  # groupA decoded=9, groupB=4
    # every decoded token participates in cross (both groups have P>=2)
    assert ctx.cross_local_idx.tolist() == list(range(13))


def test_repack_loss_fields_and_token_roundtrip():
    seqs = [
        ([10, 11, 12], [20, 21]),
        ([10, 11, 12], [22, 23, 24]),
        ([30, 31], [40, 41, 42]),
    ]
    b = _build_packed(seqs)
    out = shared_prefix_repack_batch(b)
    ctx = out["shared_prefix_context"]
    orig_ids = b["input_ids"].squeeze(0)
    rep_ids = out["input_ids"].squeeze(0)

    # decoded tokens map 1:1 from original (boundary + response)
    assert torch.equal(rep_ids[ctx.dec_idx], orig_ids[ctx.dec_orig_idx])

    # repacked loss fields: decoded carry original values, shared carry fill
    assert torch.equal(out["target_tokens"].squeeze(0)[ctx.dec_idx], b["target_tokens"].squeeze(0)[ctx.dec_orig_idx])
    assert (out["target_tokens"].squeeze(0)[ctx.shared_idx] == IGNORE).all()
    assert torch.equal(out["advantages"].squeeze(0)[ctx.dec_idx], b["advantages"].squeeze(0)[ctx.dec_orig_idx])

    # cu_seq_lens dropped (shared-prefix backend drives attention from the context)
    assert "cu_seq_lens_q" not in out


def test_remap_to_original_round_trips():
    seqs = [
        ([10, 11, 12], [20, 21]),
        ([10, 11, 12], [22, 23, 24]),
        ([30, 31], [40, 41, 42]),
    ]
    b = _build_packed(seqs)
    out = shared_prefix_repack_batch(b)
    ctx = out["shared_prefix_context"]

    # a per-token tensor in repacked order, remapped back to original layout
    rep = torch.arange(ctx.repacked_len, dtype=torch.float32).unsqueeze(0)
    orig = shared_prefix_remap_to_original(rep, ctx, fill=-1.0)
    assert orig.shape == (1, ctx.orig_len)
    # decoded positions carry their repacked index; prompt-interior get the fill
    assert torch.equal(orig.squeeze(0)[ctx.dec_orig_idx], rep.squeeze(0)[ctx.dec_idx])
    interior = [i for i in range(ctx.orig_len) if i not in set(ctx.dec_orig_idx.tolist())]
    assert all(orig.squeeze(0)[i].item() == -1.0 for i in interior)


def test_p_equals_one_group_has_empty_shared_block():
    # prompt is a single token shared by 2 members -> shared block empty, cross skipped
    seqs = [([7], [20, 21]), ([7], [22, 23])]
    b = _build_packed(seqs)
    out = shared_prefix_repack_batch(b)
    ctx = out["shared_prefix_context"]
    assert ctx.shared_idx.numel() == 0
    assert ctx.cross_local_idx.numel() == 0
    # decoded blocks [7,20,21] and [7,22,23]
    assert out["input_ids"].squeeze(0).tolist() == [7, 20, 21, 7, 22, 23]
