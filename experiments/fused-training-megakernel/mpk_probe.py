"""mpk_probe.py — MPK-topology gate: dedicated scheduler BLOCKS + full-register consumers.

The P4a/P4b ws verdict: the scheduler-offload protocol wins, but paying for it with a
384-thread block caps consumers at 224 regs — a uniform ~8-20% op tax that eats the
win. The MPK topology (Mirage Persistent Kernel, 2025; P6 survey item 7) sidesteps
the register tax: ONE scheduler block drives 131 pure-consumer blocks (256 threads,
full 255-reg budget) through per-consumer gmem mailboxes.

TWO questions, one probe:
  1. Mailbox hop cost: consumer done -> scheduler accounts -> claims -> mailbox write
     -> consumer starts. Two cross-SM signals per hop (vs df's one). Gate: <= ~3us/hop
     (the current df ring) on a serial chain of 128x64x64 NT wgmma gemm instrs (the
     ws_probe/hop_bench shape: flat interpreter 5.7, ws probe 2.8).
  2. Scheduler-issued prefetch.global.L2 of the NEXT instr's operands during the
     current instr's compute: L2 is device-wide, so a scheduler block CAN warm it for
     consumers. Measured on DRAM-cold operands (distinct buffers per instr, working
     set >> L2).

Topology: grid = 1 + NCONS blocks. Block 0 = scheduler (NCONS threads, thread i owns
consumer i's mailbox). Blocks 1..NCONS = consumers. Each chain instr has exactly
NCONS tile batches (one per consumer, claim pre-assigned = thread i gets tile i) so a
hop = all consumers finish instr k -> scheduler flips everyone to k+1. Protocol:
monotonic seq counters in gmem, ld.acquire.gpu / st.release.gpu (the ws_probe
recipe, gmem scope).

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python mpk_probe.py
"""

import statistics

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

#define BM 128
#define BN 64
#define BK 64
#define LDC 68

// ---- SW128 layout + descriptors (pipe_probe recipe) -----------------------------------
__device__ __forceinline__ int koff_sw(int r, int k8) {
  return r * 128 + ((((k8 >> 3) ^ (r & 7)) << 4));
}
__device__ __forceinline__ uint64_t desc_k_sw(const void* slab, int s) {
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(slab) + s * 32;
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = 0;
  d.bitfield.stride_byte_offset_ = (1024 >> 4);
  d.bitfield.layout_type_ = 1;
  return d.desc_;
}

__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.global.release.gpu.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
__device__ __forceinline__ long long gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// per-consumer mailbox, one 128B line each
struct __align__(128) Mailbox {
  int full_seq;   // scheduler -> consumer: instr k assigned
  int done_seq;   // consumer -> scheduler: instr k finished
  int pad[30];
};

// one chain instr = NT gemm C[i] = A[i] @ B[i]^T over M=128*NCONS rows; consumer c
// owns row band c (tile c). K = BK (single k-stage: the hop-latency regime).
// A[i]: [M, K] bf16; B[i]: [BN, K]; C[i]: [M, BN].

template <int PREFETCH>
__global__ void mpk_chain(const bf16* const* __restrict__ As,
                          const bf16* const* __restrict__ Bs,
                          bf16* const* __restrict__ Cs_, int n_instr, int ncons, int K,
                          Mailbox* __restrict__ mb, long long* __restrict__ stamps) {
  extern __shared__ char smem[];
  char* smem_al = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem) + 1023) & ~uintptr_t(1023));

  if (blockIdx.x == 0) {
    // ---------------- scheduler block: thread i owns consumer i ----------------
    const int c = threadIdx.x;
    if (c >= ncons) return;
    for (int k = 0; k < n_instr; ++k) {
      // assign instr k to consumer c
      st_release_gpu(&mb[c].full_seq, k + 1);
      if (PREFETCH && k + 1 < n_instr) {
        // warm L2 for instr k+1 while consumers execute k: consumer c's A band +
        // the shared B. 128 rows x K elts: prefetch one 128B line per 128 elts.
        const bf16* An = As[k + 1] + (int64_t)c * BM * K;
        for (int off = 0; off < BM * K; off += 64) prefetch_l2(An + off);
        const bf16* Bn = Bs[k + 1];
        for (int off = c * 64; off < BN * K; off += 64 * ncons) prefetch_l2(Bn + off);
      }
      // wait for completion of k
      while (ld_acquire_gpu(&mb[c].done_seq) <= k) __nanosleep(64);
      if (c == 0 && stamps) stamps[k] = gt();
    }
    return;
  }

  // ---------------- consumer block ----------------
  const int c = blockIdx.x - 1;
  const int tid = threadIdx.x;
  const int wg = tid / 128;
  const int wtid = tid % 128;
  bf16* Asm = reinterpret_cast<bf16*>(smem_al);           // A slabs: 2 x 8KB
  bf16* Bsm = reinterpret_cast<bf16*>(smem_al + 16384);   // B slab: 8KB
  float* Cst = reinterpret_cast<float*>(smem_al);         // epilogue overlay

  for (int k = 0; k < n_instr; ++k) {
    while (ld_acquire_gpu(&mb[c].full_seq) <= k) __nanosleep(64);
    const bf16* A = As[k] + (int64_t)c * BM * K;
    const bf16* B = Bs[k];
    bf16* C = Cs_[k] + (int64_t)c * BM * BN;
    const int iters = K / BK;
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.0f;
    for (int t = 0; t < iters; ++t) {
      // stage loads (SW128 K-major, conflict-free)
#pragma unroll
      for (int i = 0; i < 4; ++i) {  // A: 128r x 64k
        const int v = tid + i * 256;
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(Asm + (r / 64) * 4096) + koff_sw(r % 64, k8),
            &A[(int64_t)r * K + t * BK + k8], 16);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {  // B: 64r x 64k
        const int v = tid + i * 256;
        const int r = v / 8, k8 = (v % 8) * 8;
        __pipeline_memcpy_async(reinterpret_cast<char*>(Bsm) + koff_sw(r, k8),
                                &B[(int64_t)r * K + t * BK + k8], 16);
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = desc_k_sw(Asm + wg * 4096, s);
        db[s] = desc_k_sw(Bsm, s);
      }
      cute::warpgroup_arrive();
#pragma unroll
      for (int s = 0; s < 4; ++s)
        SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>::fma(
            da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
            d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
            d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
            SG::ScaleOut::One);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      __syncthreads();
    }
    // epilogue: stage + coalesced bf16 store
    const int w = wtid / 32, l = wtid % 32;
    {
      const int r = wg * 64 + w * 16 + l / 4;
      const int cb = (l % 4) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            Cst[(r + 8 * i) * LDC + n8 * 8 + cb + j] = d[n8 * 4 + i * 2 + j];
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int gid = tid + g * 256;
      const int m = gid / 8, c8 = (gid % 8) * 8;
      uint4 out_v;
      bf16* oe = reinterpret_cast<bf16*>(&out_v);
#pragma unroll
      for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(Cst[m * LDC + c8 + e]);
      *reinterpret_cast<uint4*>(&C[(int64_t)m * BN + c8]) = out_v;
    }
    __syncthreads();
    if (tid == 0) st_release_gpu(&mb[c].done_seq, k + 1);
  }
}

void run_mpk(torch::Tensor Aptrs, torch::Tensor Bptrs, torch::Tensor Cptrs,
             int64_t n_instr, int64_t ncons, int64_t K, torch::Tensor mailboxes,
             torch::Tensor stamps, int64_t prefetch) {
  const bf16* const* As = reinterpret_cast<const bf16* const*>(Aptrs.data_ptr<int64_t>());
  const bf16* const* Bs = reinterpret_cast<const bf16* const*>(Bptrs.data_ptr<int64_t>());
  bf16* const* Cs = reinterpret_cast<bf16* const*>(Cptrs.data_ptr<int64_t>());
  Mailbox* mb = reinterpret_cast<Mailbox*>(mailboxes.data_ptr());
  long long* st = reinterpret_cast<long long*>(stamps.data_ptr<int64_t>());
  const int smem = 36 * 1024;  // 24KB stages + 1KB align + headroom (Cs overlay fits)
  auto launch = [&](auto kern) {
    cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<(int)(1 + ncons), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        As, Bs, Cs, (int)n_instr, (int)ncons, (int)K, mb, st);
  };
  if (prefetch)
    launch(mpk_chain<1>);
  else
    launch(mpk_chain<0>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

cpp_src = "void run_mpk(torch::Tensor Aptrs, torch::Tensor Bptrs, torch::Tensor Cptrs, int64_t n_instr, int64_t ncons, int64_t K, torch::Tensor mailboxes, torch::Tensor stamps, int64_t prefetch);"


def main():
    torch.cuda.set_device(0)
    ext = load_inline(
        name="mpk_probe",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["run_mpk"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-I{CUTE_INC}", "--expt-relaxed-constexpr"],
        extra_include_paths=[CUTE_INC],
        verbose=False,
    )
    print("built ok")
    NCONS = 131  # 1 scheduler + 131 consumers = 132 blocks = 1/SM
    NI = 64
    K = 64  # single k-stage: hop-latency regime (the ws_probe shape)
    M = 128 * NCONS
    torch.manual_seed(0)
    # distinct operands per instr; working set = NI * (M*K + 64*K) * 2B ~= 137MB >> L2
    As = [torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for _ in range(NI)]
    Bs = [torch.randn(64, K, device="cuda", dtype=torch.bfloat16) for _ in range(NI)]
    Cs = [torch.empty(M, 64, device="cuda", dtype=torch.bfloat16) for _ in range(NI)]
    Ap = torch.tensor([t.data_ptr() for t in As], dtype=torch.int64, device="cuda")
    Bp = torch.tensor([t.data_ptr() for t in Bs], dtype=torch.int64, device="cuda")
    Cp = torch.tensor([t.data_ptr() for t in Cs], dtype=torch.int64, device="cuda")
    mb = torch.zeros(NCONS * 32, dtype=torch.int32, device="cuda")
    stamps = torch.zeros(NI, dtype=torch.int64, device="cuda")

    # correctness once
    mb.zero_()
    ext.run_mpk(Ap, Bp, Cp, NI, NCONS, K, mb, stamps, 0)
    torch.cuda.synchronize()
    ref = As[7].float() @ Bs[7].float().T
    err = (Cs[7].float() - ref).abs().max().item()
    assert err < 0.5, f"mismatch {err}"
    print(f"parity ok (max abs err {err:.3e})")

    for pf in (0, 1):
        times = []
        for _ in range(30):
            mb.zero_()
            torch.cuda.synchronize()
            # cold-ish L2 per iteration: touch a 60MB scrub buffer
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            ext.run_mpk(Ap, Bp, Cp, NI, NCONS, K, mb, stamps, pf)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3)
        med = statistics.median(times)
        # per-hop from the scheduler stamps of the last run
        st = stamps.cpu().tolist()
        hops = [(st[i + 1] - st[i]) / 1e3 for i in range(8, NI - 1)]
        hops.sort()
        print(
            f"prefetch={pf}: total {med:8.1f}us for {NI} instrs = {med / NI:5.2f}us/hop"
            f"   (stamp-median {hops[len(hops) // 2]:5.2f}us)"
        )


if __name__ == "__main__":
    main()
