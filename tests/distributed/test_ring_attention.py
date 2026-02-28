"""Tests for ring attention (context parallelism).

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/distributed/test_ring_attention.py -v
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ring_attention.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist
from flash_attn import flash_attn_func, flash_attn_varlen_func

from xorl.distributed.sequence_parallel.ring_attention import (
    _get_zigzag_step_section,
    _merge_attn_outputs,
    ring_flash_attention_forward,
)
from xorl.data.collators.sequence_shard_collator import (
    zigzag_reorder_packed_sequence,
)

pytestmark = [pytest.mark.distributed]


def is_distributed_available():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def requires_distributed(func):
    return pytest.mark.skipif(
        not is_distributed_available(),
        reason="Test requires distributed environment (run with torchrun)",
    )(func)


def requires_min_gpus(n):
    def decorator(func):
        return pytest.mark.skipif(
            not is_distributed_available() or int(os.environ.get("WORLD_SIZE", 0)) < n,
            reason=f"Test requires at least {n} GPUs",
        )(func)
    return decorator


def setup_dist():
    """Initialize distributed if not already done."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


# ------------------------------------------------------------------ #
# Unit tests (no distributed required)
# ------------------------------------------------------------------ #



class TestLSEMerge:
    """Test _merge_attn_outputs numerical stability."""

    def test_basic_merge_batched(self):
        """Merge two partial outputs and verify against manual computation."""
        B, S, H, D = 1, 8, 4, 32
        torch.manual_seed(42)
        out1 = torch.randn(B, S, H, D, device="cuda")
        out2 = torch.randn(B, S, H, D, device="cuda")
        lse1 = torch.randn(B, H, S, device="cuda")
        lse2 = torch.randn(B, H, S, device="cuda")

        merged_out, merged_lse = _merge_attn_outputs(out1, lse1, out2, lse2, is_varlen=False)

        # Manual reference: merged = (exp(lse1) * out1 + exp(lse2) * out2) / (exp(lse1) + exp(lse2))
        w1 = torch.exp(lse1 - merged_lse).transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]
        w2 = torch.exp(lse2 - merged_lse).transpose(1, 2).unsqueeze(-1)
        ref_out = w1 * out1 + w2 * out2

        assert torch.allclose(merged_out, ref_out, atol=1e-5)

    def test_basic_merge_varlen(self):
        """Merge two partial outputs in varlen layout."""
        total, H, D = 16, 4, 32
        torch.manual_seed(42)
        out1 = torch.randn(total, H, D, device="cuda")
        out2 = torch.randn(total, H, D, device="cuda")
        lse1 = torch.randn(H, total, device="cuda")
        lse2 = torch.randn(H, total, device="cuda")

        merged_out, merged_lse = _merge_attn_outputs(out1, lse1, out2, lse2, is_varlen=True)

        assert merged_out.shape == (total, H, D)
        assert merged_lse.shape == (H, total)
        assert not torch.isnan(merged_out).any()
        assert not torch.isnan(merged_lse).any()

    def test_extreme_lse_values(self):
        """Merge should be stable with extreme LSE differences."""
        B, S, H, D = 1, 4, 2, 16
        out1 = torch.randn(B, S, H, D, device="cuda")
        out2 = torch.randn(B, S, H, D, device="cuda")
        # Large LSE difference — one dominates
        lse1 = torch.full((B, H, S), 100.0, device="cuda")
        lse2 = torch.full((B, H, S), -100.0, device="cuda")

        merged_out, merged_lse = _merge_attn_outputs(out1, lse1, out2, lse2, is_varlen=False)

        # out1 should dominate since its lse is much larger
        assert torch.allclose(merged_out, out1, atol=1e-4)
        assert not torch.isnan(merged_out).any()
        assert not torch.isinf(merged_out).any()

    def test_equal_lse(self):
        """When LSE values are equal, output should be average."""
        B, S, H, D = 1, 4, 2, 16
        out1 = torch.randn(B, S, H, D, device="cuda")
        out2 = torch.randn(B, S, H, D, device="cuda")
        lse = torch.randn(B, H, S, device="cuda")

        merged_out, _ = _merge_attn_outputs(out1, lse.clone(), out2, lse.clone(), is_varlen=False)
        expected = (out1 + out2) / 2
        assert torch.allclose(merged_out, expected, atol=1e-5)


# ------------------------------------------------------------------ #
# Distributed tests
# ------------------------------------------------------------------ #


@requires_distributed
class TestRingAttentionNonCausal:
    """Test ring attention for non-causal (no zigzag needed)."""

    def test_ring_attention_non_causal(self):
        """Ring attention (non-causal) matches single-GPU flash attention."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        B, S_full, H, D = 1, 16 * world_size, 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")

        ref_out = flash_attn_func(q_full, k_full, v_full, causal=False)

        S_local = S_full // world_size
        q_local = q_full[:, rank * S_local : (rank + 1) * S_local].contiguous()
        k_local = k_full[:, rank * S_local : (rank + 1) * S_local].contiguous()
        v_local = v_full[:, rank * S_local : (rank + 1) * S_local].contiguous()

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group,
            causal=False,
        )

        ref_local = ref_out[:, rank * S_local : (rank + 1) * S_local]
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)


@requires_min_gpus(4)
class TestHybridUlyssesRing:
    """Test hybrid Ulysses + Ring attention (requires 4+ GPUs)."""

    def test_hybrid_forward(self):
        """Hybrid Ulysses+Ring matches single-GPU flash attention."""
        rank, world_size = setup_dist()
        assert world_size >= 4, "Need at least 4 GPUs"

        ulysses_size = 2
        cp_size = world_size // ulysses_size

        from xorl.distributed.sequence_parallel.comm import (
            init_sequence_parallel,
            get_context_parallel_group,
            get_ulysses_sequence_parallel_group,
        )

        init_sequence_parallel(ulysses_size=ulysses_size, cp_size=cp_size, sep_dp=False)

        cp_group = get_context_parallel_group()
        ulysses_group = get_ulysses_sequence_parallel_group()

        B, S_full, H, D = 1, 64, 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")

        # Reference
        ref_out = flash_attn_func(q_full, k_full, v_full, causal=True)

        # Hybrid: zigzag reorder then split by sp_size, then Ulysses a2a gathers
        sp_size = ulysses_size * cp_size
        S_local = S_full // sp_size

        # Zigzag reorder for causal ring attention (matches collator behavior)
        position_ids = torch.arange(S_full, device="cuda").unsqueeze(0)
        q_zz = zigzag_reorder_packed_sequence(q_full, position_ids, cp_size, dim=1)
        k_zz = zigzag_reorder_packed_sequence(k_full, position_ids, cp_size, dim=1)
        v_zz = zigzag_reorder_packed_sequence(v_full, position_ids, cp_size, dim=1)

        sp_rank = rank
        q_local = q_zz[:, sp_rank * S_local : (sp_rank + 1) * S_local].contiguous()
        k_local = k_zz[:, sp_rank * S_local : (sp_rank + 1) * S_local].contiguous()
        v_local = v_zz[:, sp_rank * S_local : (sp_rank + 1) * S_local].contiguous()

        from xorl.distributed.sequence_parallel.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )

        # Pre-attention Ulysses a2a (gather seq, scatter heads)
        q_a2a = gather_seq_scatter_heads(q_local.squeeze(0), seq_dim=0, head_dim=1, group=ulysses_group).unsqueeze(0)
        k_a2a = gather_seq_scatter_heads(k_local.squeeze(0), seq_dim=0, head_dim=1, group=ulysses_group).unsqueeze(0)
        v_a2a = gather_seq_scatter_heads(v_local.squeeze(0), seq_dim=0, head_dim=1, group=ulysses_group).unsqueeze(0)

        # Ring attention across CP group
        ring_out = ring_flash_attention_forward(
            q_a2a, k_a2a, v_a2a,
            cp_group=cp_group,
            causal=True,
        )

        # Post-attention Ulysses a2a (gather heads, scatter seq)
        out_a2a = gather_heads_scatter_seq(ring_out.squeeze(0), seq_dim=0, head_dim=1, group=ulysses_group).unsqueeze(0)

        # Compare with zigzag-reordered reference
        ref_out_zz = zigzag_reorder_packed_sequence(ref_out, position_ids, cp_size, dim=1)
        ref_local = ref_out_zz[:, sp_rank * S_local : (sp_rank + 1) * S_local]
        torch.testing.assert_close(out_a2a, ref_local, atol=1e-2, rtol=1e-2)



@requires_min_gpus(4)
class TestHybridStrategyEndToEnd:
    """End-to-end test of HybridUlyssesRingStrategy through the attention module.

    Tests output correctness, input gradients, and weight gradients
    (qkv_proj, o_proj, q_norm, k_norm) against a single-GPU reference.
    """

    def _setup_sp_groups(self, ulysses_size, cp_size):
        """Init SP groups if not already initialized."""
        from xorl.distributed.sequence_parallel.comm import (
            get_context_parallel_group,
            init_sequence_parallel,
        )

        if get_context_parallel_group(check_initialized=False) is None:
            init_sequence_parallel(
                ulysses_size=ulysses_size, cp_size=cp_size, sep_dp=False
            )

    def test_hybrid_strategy_output_and_gradients(self):
        """HybridUlyssesRingStrategy produces correct output, input grad, and weight grads."""
        rank, world_size = setup_dist()
        assert world_size >= 4

        ulysses_size = 2
        cp_size = world_size // ulysses_size
        sp_size = ulysses_size * cp_size

        self._setup_sp_groups(ulysses_size, cp_size)

        from xorl.distributed.sequence_parallel.comm import (
            get_context_parallel_group,
            get_ulysses_sequence_parallel_group,
            get_unified_sequence_parallel_group,
        )
        from xorl.distributed.sequence_parallel.strategy import (
            HybridUlyssesRingStrategy,
            NoopStrategy,
        )
        from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
        from xorl.models.layers.attention.multi_head_attention import MultiHeadAttention
        from xorl.models.layers.rope import RotaryEmbedding

        cp_group = get_context_parallel_group()
        ulysses_group = get_ulysses_sequence_parallel_group()
        unified_group = get_unified_sequence_parallel_group()

        # Small config for testing
        config = Qwen3Config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=32,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
        )
        config._attn_implementation = "flash_attention_2"

        # Create module with identical weights on all ranks
        torch.manual_seed(42)
        module = MultiHeadAttention(config, layer_idx=0).to(torch.bfloat16).cuda()
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

        # Create RoPE
        rope = RotaryEmbedding(config, device="cuda")

        B = 1
        S_full = 64
        S_local = S_full // sp_size

        # Create full inputs (same on all ranks via same seed)
        torch.manual_seed(123)
        hidden_full = torch.randn(
            B, S_full, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        position_ids_full = torch.arange(S_full, device="cuda").unsqueeze(0)
        cos_full, sin_full = rope(hidden_full, position_ids_full)

        # --- Reference: NoopStrategy on full sequence ---
        hidden_ref = hidden_full.clone().requires_grad_(True)

        noop = NoopStrategy()
        q_ref, k_ref, v_ref = noop.project_qkv(
            module, hidden_ref, (cos_full.clone(), sin_full.clone())
        )
        out_ref = noop.compute_attention(module, q_ref, k_ref, v_ref, None)
        out_ref = noop.project_output(module, out_ref)
        out_ref.sum().backward()

        ref_hidden_grad = hidden_ref.grad.clone()
        ref_qkv_weight_grad = module.qkv_proj.weight.grad.clone()
        ref_o_weight_grad = module.o_proj.weight.grad.clone()
        ref_qnorm_grad = module.q_norm.weight.grad.clone()
        ref_knorm_grad = module.k_norm.weight.grad.clone()

        # Zero grads for distributed run
        module.zero_grad()

        # --- Distributed: HybridUlyssesRingStrategy ---
        # Causal ring attention expects zigzag-reordered data (as the collator
        # would produce). Zigzag reorder hidden_states and position embeddings
        # before SP slicing, matching the real data pipeline.
        strategy = HybridUlyssesRingStrategy(
            ulysses_group, cp_group, ulysses_size
        )

        hidden_zz = zigzag_reorder_packed_sequence(
            hidden_full, position_ids_full, cp_size, dim=1
        )
        cos_zz = zigzag_reorder_packed_sequence(
            cos_full, position_ids_full, cp_size, dim=1
        )
        sin_zz = zigzag_reorder_packed_sequence(
            sin_full, position_ids_full, cp_size, dim=1
        )

        hidden_local = (
            hidden_zz[:, rank * S_local : (rank + 1) * S_local]
            .clone()
            .contiguous()
            .requires_grad_(True)
        )
        cos_local = cos_zz[:, rank * S_local : (rank + 1) * S_local].contiguous()
        sin_local = sin_zz[:, rank * S_local : (rank + 1) * S_local].contiguous()

        q_dist, k_dist, v_dist = strategy.project_qkv(
            module, hidden_local, (cos_local, sin_local)
        )
        out_dist = strategy.compute_attention(module, q_dist, k_dist, v_dist, None)
        out_dist = strategy.project_output(module, out_dist)
        out_dist.sum().backward()

        # Tolerances scale with sp_size: more ring steps + a2a accumulate
        # bfloat16 numerical error. Weight grads need larger tolerance
        # because error accumulates through the full pipeline.
        act_tol = 2e-2 * sp_size
        weight_tol = 5e-2 * sp_size

        # Zigzag-reorder reference outputs for comparison
        ref_out_zz = zigzag_reorder_packed_sequence(
            out_ref.detach(), position_ids_full, cp_size, dim=1
        )
        ref_hidden_grad_zz = zigzag_reorder_packed_sequence(
            ref_hidden_grad, position_ids_full, cp_size, dim=1
        )

        # 1. Compare output
        ref_out_local = ref_out_zz[:, rank * S_local : (rank + 1) * S_local]
        torch.testing.assert_close(
            out_dist.detach(), ref_out_local, atol=act_tol, rtol=act_tol
        )

        # 2. Compare input gradient
        ref_hidden_grad_local = ref_hidden_grad_zz[
            :, rank * S_local : (rank + 1) * S_local
        ]
        torch.testing.assert_close(
            hidden_local.grad, ref_hidden_grad_local, atol=act_tol, rtol=act_tol
        )

        # 3. Compare weight gradients (all-reduce across SP group first)
        qkv_grad = module.qkv_proj.weight.grad.clone()
        o_grad = module.o_proj.weight.grad.clone()
        qnorm_grad = module.q_norm.weight.grad.clone()
        knorm_grad = module.k_norm.weight.grad.clone()

        dist.all_reduce(qkv_grad, group=unified_group)
        dist.all_reduce(o_grad, group=unified_group)
        dist.all_reduce(qnorm_grad, group=unified_group)
        dist.all_reduce(knorm_grad, group=unified_group)

        torch.testing.assert_close(
            qkv_grad, ref_qkv_weight_grad, atol=weight_tol, rtol=weight_tol
        )
        torch.testing.assert_close(
            o_grad, ref_o_weight_grad, atol=weight_tol, rtol=weight_tol
        )
        torch.testing.assert_close(
            qnorm_grad, ref_qnorm_grad, atol=weight_tol, rtol=weight_tol
        )
        torch.testing.assert_close(
            knorm_grad, ref_knorm_grad, atol=weight_tol, rtol=weight_tol
        )


# ------------------------------------------------------------------ #
# Load balancing tests
# ------------------------------------------------------------------ #


class TestZigzagUnit:
    """Unit tests for Zigzag section logic (no GPU needed)."""

    def test_all_ranks_compute_all_steps(self):
        """With load balancing, every rank computes every step."""
        for cp_size in [2, 4, 8]:
            for rank in range(cp_size):
                sections = [_get_zigzag_step_section(rank, cp_size, s) for s in range(cp_size)]
                assert len(sections) == cp_size, f"rank={rank}, cp_size={cp_size}"
                assert sections[0] == "diagonal"

    def test_section_types(self):
        """Verify section types for cp_size=4."""
        # All non-diagonal sections should be either "lower" or "upper"
        for rank in range(4):
            for step in range(1, 4):
                section = _get_zigzag_step_section(rank, 4, step)
                assert section in ("lower", "upper"), \
                    f"rank={rank}, step={step}, section={section}"


class TestZigzagReorderUnit:
    """Unit tests for zigzag_reorder_packed_sequence (CPU, no GPU needed)."""

    def test_single_doc_reorder(self):
        """Single document reorder produces correct sub-chunk ordering."""
        cp_size = 2
        doc_len = 40  # 4 sub-chunks of 10
        tensor = torch.arange(doc_len).unsqueeze(0)  # [1, 40]
        position_ids = torch.arange(doc_len).unsqueeze(0)  # [1, 40]

        reordered = zigzag_reorder_packed_sequence(tensor, position_ids, cp_size, dim=-1)

        # Single doc: rank 0 gets [s0, s3], rank 1 gets [s1, s2]
        # = [0-9, 30-39, 10-19, 20-29]
        expected = torch.cat([
            torch.arange(0, 10),
            torch.arange(30, 40),
            torch.arange(10, 20),
            torch.arange(20, 30),
        ]).unsqueeze(0)
        assert torch.equal(reordered, expected)

    def test_multi_doc_reorder(self):
        """Multiple packed documents: rank-first grouping for contiguous sp_slice."""
        cp_size = 2
        doc_len = 20  # 4 sub-chunks of 5
        total = 2 * doc_len
        tensor = torch.arange(total).unsqueeze(0)  # [1, 40]
        position_ids = torch.cat([
            torch.arange(doc_len), torch.arange(doc_len)
        ]).unsqueeze(0)

        reordered = zigzag_reorder_packed_sequence(tensor, position_ids, cp_size, dim=-1)

        # Rank 0's data: doc0_s0, doc0_s3, doc1_s0, doc1_s3
        # Rank 1's data: doc0_s1, doc0_s2, doc1_s1, doc1_s2
        expected = torch.cat([
            # Rank 0
            torch.arange(0, 5), torch.arange(15, 20),    # doc0: s0, s3
            torch.arange(20, 25), torch.arange(35, 40),  # doc1: s0, s3
            # Rank 1
            torch.arange(5, 10), torch.arange(10, 15),   # doc0: s1, s2
            torch.arange(25, 30), torch.arange(30, 35),  # doc1: s1, s2
        ]).unsqueeze(0)
        assert torch.equal(reordered, expected)

    def test_position_ids_reorder(self):
        """Position IDs after zigzag reorder give correct cu_seqlens per rank."""
        cp_size = 2
        doc_len = 20
        num_docs = 2
        total = doc_len * num_docs
        position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)

        reordered_pos = zigzag_reorder_packed_sequence(
            position_ids, position_ids, cp_size, dim=-1
        )

        # Rank 0's slice (first half): should have 2 docs, each starting at pos 0
        half = total // 2
        rank0_pos = reordered_pos[0, :half]
        # Doc boundaries: position 0 appears at start of each doc's sub-chunks
        zeros = (rank0_pos == 0).nonzero(as_tuple=False).view(-1).tolist()
        assert len(zeros) == 2, f"Expected 2 doc boundaries, got {zeros}"

    def test_various_cp_sizes(self):
        """Test zigzag reorder for cp_size=2,4,8."""
        for cp_size in [2, 4, 8]:
            n = 2 * cp_size
            doc_len = n * 4  # 4 tokens per sub-chunk
            tensor = torch.arange(doc_len).unsqueeze(0)
            position_ids = torch.arange(doc_len).unsqueeze(0)

            reordered = zigzag_reorder_packed_sequence(
                tensor, position_ids, cp_size, dim=-1
            )

            # Verify shape preserved
            assert reordered.shape == tensor.shape
            # Verify it's a permutation
            assert set(reordered[0].tolist()) == set(tensor[0].tolist())
            # Verify each rank's contiguous slice has [early, late] sub-chunks
            chunk_size = doc_len // cp_size
            for r in range(cp_size):
                rank_slice = reordered[0, r * chunk_size : (r + 1) * chunk_size]
                sub_size = chunk_size // 2
                early = rank_slice[:sub_size]
                late = rank_slice[sub_size:]
                # Early sub-chunk should have lower values than late
                assert early.max() < late.min(), \
                    f"cp_size={cp_size}, rank={r}: early max={early.max()} >= late min={late.min()}"

    def test_cp_size_1_noop(self):
        """cp_size=1 should return the original tensor unchanged."""
        tensor = torch.arange(20).unsqueeze(0)
        position_ids = torch.arange(20).unsqueeze(0)
        result = zigzag_reorder_packed_sequence(tensor, position_ids, 1, dim=-1)
        assert torch.equal(result, tensor)

    def test_invalid_length_raises(self):
        """Document length not divisible by 2*cp_size should raise ValueError."""
        cp_size = 2
        doc_len = 15  # not divisible by 4
        tensor = torch.arange(doc_len).unsqueeze(0)
        position_ids = torch.arange(doc_len).unsqueeze(0)
        with pytest.raises(ValueError, match="not divisible"):
            zigzag_reorder_packed_sequence(tensor, position_ids, cp_size, dim=-1)


@requires_distributed
class TestZigzagDistributed:
    """Test zigzag ring attention with data-pipeline zigzag reorder."""

    def _zigzag_split(self, full_tensor, position_ids, cp_size, cp_rank, dim=-1):
        """Apply zigzag reorder then take cp_rank's contiguous slice."""
        reordered = zigzag_reorder_packed_sequence(full_tensor, position_ids, cp_size, dim=dim)
        seq_len = reordered.shape[dim]
        chunk = seq_len // cp_size
        return reordered.narrow(dim, cp_rank * chunk, chunk).contiguous()

    def test_zigzag_varlen_forward(self):
        """Varlen zigzag ring attention matches single-GPU reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        num_docs = 2
        doc_len = 16 * world_size  # divisible by 2*cp_size
        total = doc_len * num_docs
        H, D = 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        cu_seqlens_full = torch.tensor(
            [i * doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        # Reference: single-GPU
        ref_out = flash_attn_varlen_func(
            q_full, k_full, v_full,
            cu_seqlens_full, cu_seqlens_full,
            doc_len, doc_len, causal=True,
        )

        # Zigzag reorder then split per rank (varlen: dim=0)
        position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
        q_local = self._zigzag_split(q_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        k_local = self._zigzag_split(k_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        v_local = self._zigzag_split(v_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)

        local_doc_len = doc_len // world_size
        cu_seqlens_local = torch.tensor(
            [i * local_doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
            cu_seqlens_q=cu_seqlens_local, cu_seqlens_k=cu_seqlens_local,
            max_seqlen_q=local_doc_len, max_seqlen_k=local_doc_len,
        )

        # Compare: reorder reference output the same way
        ref_local = self._zigzag_split(ref_out.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)

    def test_zigzag_varlen_backward(self):
        """Varlen zigzag ring attention backward gradients match reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        num_docs = 2
        doc_len = 16 * world_size
        total = doc_len * num_docs
        H, D = 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        cu_seqlens_full = torch.tensor(
            [i * doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        ref_out = flash_attn_varlen_func(
            q_full, k_full, v_full,
            cu_seqlens_full, cu_seqlens_full,
            doc_len, doc_len, causal=True,
        )
        ref_out.sum().backward()
        ref_dq = q_full.grad.clone()
        ref_dk = k_full.grad.clone()
        ref_dv = v_full.grad.clone()

        # Zigzag split
        position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
        q_local = self._zigzag_split(q_full.detach().unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0).requires_grad_(True)
        k_local = self._zigzag_split(k_full.detach().unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0).requires_grad_(True)
        v_local = self._zigzag_split(v_full.detach().unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0).requires_grad_(True)

        local_doc_len = doc_len // world_size
        cu_seqlens_local = torch.tensor(
            [i * local_doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
            cu_seqlens_q=cu_seqlens_local, cu_seqlens_k=cu_seqlens_local,
            max_seqlen_q=local_doc_len, max_seqlen_k=local_doc_len,
        )
        ring_out.sum().backward()

        # Compare gradients (zigzag-reordered reference)
        tol = 1e-2 * world_size
        ref_dq_local = self._zigzag_split(ref_dq.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        ref_dk_local = self._zigzag_split(ref_dk.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        ref_dv_local = self._zigzag_split(ref_dv.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)

        torch.testing.assert_close(q_local.grad, ref_dq_local, atol=tol, rtol=tol)
        torch.testing.assert_close(k_local.grad, ref_dk_local, atol=tol, rtol=tol)
        torch.testing.assert_close(v_local.grad, ref_dv_local, atol=tol, rtol=tol)

    def test_zigzag_varlen_gqa(self):
        """Varlen zigzag with GQA (H_q != H_kv) matches reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        num_docs = 2
        doc_len = 16 * world_size
        total = doc_len * num_docs
        H_q, H_kv, D = 8, 2, 64
        torch.manual_seed(42)

        q_full = torch.randn(total, H_q, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(total, H_kv, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(total, H_kv, D, dtype=torch.bfloat16, device="cuda")
        cu_seqlens_full = torch.tensor(
            [i * doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        ref_out = flash_attn_varlen_func(
            q_full, k_full, v_full,
            cu_seqlens_full, cu_seqlens_full,
            doc_len, doc_len, causal=True,
        )

        position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
        q_local = self._zigzag_split(q_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        k_local = self._zigzag_split(k_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        v_local = self._zigzag_split(v_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)

        local_doc_len = doc_len // world_size
        cu_seqlens_local = torch.tensor(
            [i * local_doc_len for i in range(num_docs + 1)], dtype=torch.int32, device="cuda"
        )

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
            cu_seqlens_q=cu_seqlens_local, cu_seqlens_k=cu_seqlens_local,
            max_seqlen_q=local_doc_len, max_seqlen_k=local_doc_len,
        )

        ref_local = self._zigzag_split(ref_out.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)

    def test_zigzag_varlen_single_doc(self):
        """Single long document with zigzag matches reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        doc_len = 32 * world_size
        H, D = 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(doc_len, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(doc_len, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(doc_len, H, D, dtype=torch.bfloat16, device="cuda")
        cu_seqlens_full = torch.tensor([0, doc_len], dtype=torch.int32, device="cuda")

        ref_out = flash_attn_varlen_func(
            q_full, k_full, v_full,
            cu_seqlens_full, cu_seqlens_full,
            doc_len, doc_len, causal=True,
        )

        position_ids = torch.arange(doc_len).unsqueeze(0)
        q_local = self._zigzag_split(q_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        k_local = self._zigzag_split(k_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        v_local = self._zigzag_split(v_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)

        local_doc_len = doc_len // world_size
        cu_seqlens_local = torch.tensor([0, local_doc_len], dtype=torch.int32, device="cuda")

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
            cu_seqlens_q=cu_seqlens_local, cu_seqlens_k=cu_seqlens_local,
            max_seqlen_q=local_doc_len, max_seqlen_k=local_doc_len,
        )

        ref_local = self._zigzag_split(ref_out.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)

    def test_zigzag_varlen_varying_doc_lengths(self):
        """Different-length documents in packed sequence with zigzag."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        n = 2 * world_size
        doc_lens = [n * 4, n * 8]  # two docs of different lengths
        total = sum(doc_lens)
        H, D = 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        cu_seqlens_full = torch.tensor(
            [0, doc_lens[0], doc_lens[0] + doc_lens[1]], dtype=torch.int32, device="cuda"
        )

        ref_out = flash_attn_varlen_func(
            q_full, k_full, v_full,
            cu_seqlens_full, cu_seqlens_full,
            max(doc_lens), max(doc_lens), causal=True,
        )

        # Build position_ids for packed sequence
        position_ids = torch.cat([torch.arange(dl) for dl in doc_lens]).unsqueeze(0)
        q_local = self._zigzag_split(q_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        k_local = self._zigzag_split(k_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        v_local = self._zigzag_split(v_full.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)

        # Local cu_seqlens: each doc_len // world_size
        local_doc_lens = [dl // world_size for dl in doc_lens]
        cu_seqlens_local = torch.tensor(
            [0] + [sum(local_doc_lens[:i+1]) for i in range(len(local_doc_lens))],
            dtype=torch.int32, device="cuda"
        )

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
            cu_seqlens_q=cu_seqlens_local, cu_seqlens_k=cu_seqlens_local,
            max_seqlen_q=max(local_doc_lens), max_seqlen_k=max(local_doc_lens),
        )

        ref_local = self._zigzag_split(ref_out.unsqueeze(0), position_ids, world_size, rank, dim=1).squeeze(0)
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)

    def test_zigzag_batched_forward(self):
        """Batched zigzag causal forward matches reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        B, S_full, H, D = 1, 64 * world_size, 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda")

        ref_out = flash_attn_func(q_full, k_full, v_full, causal=True)

        # Zigzag reorder then split
        position_ids = torch.arange(S_full).unsqueeze(0)
        S_local = S_full // world_size
        q_local = self._zigzag_split(q_full, position_ids, world_size, rank, dim=1)
        k_local = self._zigzag_split(k_full, position_ids, world_size, rank, dim=1)
        v_local = self._zigzag_split(v_full, position_ids, world_size, rank, dim=1)

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
        )

        ref_local = self._zigzag_split(ref_out, position_ids, world_size, rank, dim=1)
        torch.testing.assert_close(ring_out, ref_local, atol=1e-2, rtol=1e-2)

    def test_zigzag_batched_backward(self):
        """Batched zigzag backward gradients match reference."""
        rank, world_size = setup_dist()
        cp_group = dist.group.WORLD

        B, S_full, H, D = 1, 64 * world_size, 8, 64
        torch.manual_seed(42)

        q_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v_full = torch.randn(B, S_full, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        ref_out = flash_attn_func(q_full, k_full, v_full, causal=True)
        ref_out.sum().backward()
        ref_dq = q_full.grad.clone()
        ref_dk = k_full.grad.clone()
        ref_dv = v_full.grad.clone()

        position_ids = torch.arange(S_full).unsqueeze(0)
        S_local = S_full // world_size
        q_local = self._zigzag_split(q_full.detach(), position_ids, world_size, rank, dim=1).requires_grad_(True)
        k_local = self._zigzag_split(k_full.detach(), position_ids, world_size, rank, dim=1).requires_grad_(True)
        v_local = self._zigzag_split(v_full.detach(), position_ids, world_size, rank, dim=1).requires_grad_(True)

        ring_out = ring_flash_attention_forward(
            q_local, k_local, v_local,
            cp_group=cp_group, causal=True,
        )
        ring_out.sum().backward()

        tol = 1e-2 * world_size
        ref_dq_local = self._zigzag_split(ref_dq, position_ids, world_size, rank, dim=1)
        ref_dk_local = self._zigzag_split(ref_dk, position_ids, world_size, rank, dim=1)
        ref_dv_local = self._zigzag_split(ref_dv, position_ids, world_size, rank, dim=1)

        torch.testing.assert_close(q_local.grad, ref_dq_local, atol=tol, rtol=tol)
        torch.testing.assert_close(k_local.grad, ref_dk_local, atol=tol, rtol=tol)
        torch.testing.assert_close(v_local.grad, ref_dv_local, atol=tol, rtol=tol)
