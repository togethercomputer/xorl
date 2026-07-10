"""ws_probe.py — Phase 2 gate: warp-specialized paged-smem interpreter probe.

ONE question: can a warp-specialized (WG0 producer + WG1/2 consumers) paged-smem
interpreter execute a serial chain of wgmma GEMM instructions (128x64x64 NT, the
hop_bench.py shape that costs 5.6us/hop in the flat interpreter) at <=4us/hop?

Standalone: does not touch megakernel.cu / ops.cuh / mk.py. Reuses the validated
recipes: no-swizzle INTER descriptors (K-major SBO=256B/LBO=128B, MN-major
SBO=128B/LBO=1024B), m64n64k16 wgmma with branch-free ScaleOut::One over zeroed
accumulators, smem-staged coalesced epilogue (all from ops.cuh / wgmma_probe.py).

Run: CUDA_VISIBLE_DEVICES=3 .venv-fa4/bin/python ws_probe.py [debug|full]
"""

import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline


CUTE_INC = "/home/apanda/xorl-internal/.venv/lib/python3.12/site-packages/deep_gemm/include"

cuda_src = r"""
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

using bf16 = __nv_bfloat16;
namespace SG = cute::SM90::GMMA;

// ---- INTER (no-swizzle) smem arrangements, validated in wgmma_probe.py/ops.cuh ------
// K-major 64-row step block: offset(r,k) = (r/8)*256 + (k/8)*128 + (r%8)*16 + (k%8)*2
__device__ __forceinline__ int ws_koff(int r, int k) {
  return ((r >> 3) << 8) + ((k >> 3) << 7) + ((r & 7) << 4) + ((k & 7) << 1);
}
// MN-major 64-row step block: offset(mn,k) = (mn/8)*128 + (k/8)*1024 + (mn%8)*2 + (k%8)*16
__device__ __forceinline__ int ws_mnoff(int mn, int k) {
  return ((mn >> 3) << 7) + ((k >> 3) << 10) + ((mn & 7) << 1) + ((k & 7) << 4);
}
__device__ __forceinline__ uint64_t ws_desc(const void* p) {  // K-major INTER
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (128 >> 4);
  d.bitfield.stride_byte_offset_ = (256 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}
__device__ __forceinline__ uint64_t ws_desc_mn(const void* p) {  // MN-major INTER
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  cute::GmmaDescriptor d;
  d.desc_ = 0;
  d.bitfield.start_address_ = (addr >> 4);
  d.bitfield.leading_byte_offset_ = (1024 >> 4);
  d.bitfield.stride_byte_offset_ = (128 >> 4);
  d.bitfield.layout_type_ = 0;
  return d.desc_;
}

template <class MMA>
__device__ __forceinline__ void ws_ktile(const uint64_t (&da)[4], const uint64_t (&db)[4],
                                         float (&d)[32]) {
  cute::warpgroup_arrive();
#pragma unroll
  for (int s = 0; s < 4; ++s)
    MMA::fma(da[s], db[s], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
             d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
             d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
             SG::ScaleOut::One);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
}

// ---- mbarrier PTX helpers -----------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* m, uint32_t count) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* m) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* m, uint32_t parity) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(m);
  asm volatile(
      "{\n\t"
      ".reg .pred P;\n\t"
      "WS_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@P bra.uni WS_DONE;\n\t"
      "bra.uni WS_WAIT;\n\t"
      "WS_DONE:\n\t"
      "}\n" ::"r"(a), "r"(parity)
      : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.global.release.gpu.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ long long ws_globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---- smem layout ----------------------------------------------------------------------
// Page holds one instruction's operands: A[128,64] as 2 row-halves x 4 k16 INTER blocks,
// B[64,64] as 4 k16 INTER blocks. K=64 = a single WG_BK tile -> single stage per instr.
struct Page {
  bf16 A[2][4][1024];  // 16KB
  bf16 B[4][1024];     // 8KB
};
struct __align__(128) Ctrl {
  uint64_t full[2];   // producer(128) -> consumers
  uint64_t empty[2];  // consumers(256) -> producer
  struct __align__(16) Slot {
    unsigned long long cptr;
    int idx;
    int mode;  // 0 = NT (K,K), 1 = NN (K,MN; store C transposed), 2 = TN (MN,K)
  } slot[2];
  int claim[2];
};
#define WS_LDC 68
#define WS_CTRL_BYTES 128
#define WS_SMEM_USED (WS_CTRL_BYTES + 2 * (int)sizeof(Page) + 128 * WS_LDC * 4)
#define WS_SMEM_REQ (120 * 1024)  // pad: force 1 CTA/SM (2 CTAs + setmaxnreg deadlocks)

// ---- producer-side cp.async loaders (recipes = ops.cuh issue_stage) -------------------
__device__ __forceinline__ void issue_A(Page& P, const bf16* A, int md, int wtid) {
  if (md != 2) {  // A[128,64] K-contiguous -> K-major blocks
#pragma unroll
    for (int t = 0; t < 8; ++t) {  // 1024 16B vectors / 128 threads
      const int v = wtid + t * 128;
      const int r = v >> 3, k8 = (v & 7) << 3;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(P.A[r >> 6][k8 >> 4]) + ws_koff(r & 63, k8 & 15),
          A + r * 64 + k8, 16);
    }
  } else {  // A stored [64,128] M-contiguous -> MN-major blocks
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int v = wtid + t * 128;
      const int h = v >> 9, w_ = v & 511;
      const int k = w_ >> 3, m8 = (w_ & 7) << 3;
      __pipeline_memcpy_async(
          reinterpret_cast<char*>(P.A[h][k >> 4]) + ws_mnoff(m8, k & 15),
          A + k * 128 + h * 64 + m8, 16);
    }
  }
}
__device__ __forceinline__ void issue_B(Page& P, const bf16* B, int md, int wtid) {
  if (md != 1) {  // B[64,64] K-contiguous -> K-major blocks
#pragma unroll
    for (int t = 0; t < 4; ++t) {  // 512 16B vectors / 128 threads
      const int v = wtid + t * 128;
      const int r = v >> 3, k8 = (v & 7) << 3;
      __pipeline_memcpy_async(reinterpret_cast<char*>(P.B[k8 >> 4]) + ws_koff(r, k8 & 15),
                              B + r * 64 + k8, 16);
    }
  } else {  // B stored [64,64] N-contiguous ([K,N]) -> MN-major blocks
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int v = wtid + t * 128;
      const int k = v >> 3, n8 = (v & 7) << 3;
      __pipeline_memcpy_async(reinterpret_cast<char*>(P.B[k >> 4]) + ws_mnoff(n8, k & 15),
                              B + k * 64 + n8, 16);
    }
  }
}

// ---- the warp-specialized interpreter kernel ------------------------------------------
// __maxnreg__(168): ptxas needs the entry register count to honor setmaxnreg
// (else "C7508 setmaxnreg ignored"). Feasibility: 128*(168-DEC) >= 256*(224-168)
// -> DEC <= 56; 384*168 = 64512 <= 64K. (__launch_bounds__ cannot combine with
// __maxnreg__.) DEC in {40, 56}: 40 spills the producer path (STACK 152).
template <int SMR>  // 0 = off, else the WG0 dec target
__global__ void __maxnreg__(168) ws_kernel(
    const unsigned long long* __restrict__ Aptrs,
    const unsigned long long* __restrict__ Bptrs,
    const unsigned long long* __restrict__ Cptrs, const int* __restrict__ modes,
    int n_instr, int* done, int* cursor, long long* stamps, int prefetch_b, int done_by,
    int handoff) {
  extern __shared__ char smem[];
  Ctrl* C = reinterpret_cast<Ctrl*>(smem);
  Page* pages = reinterpret_cast<Page*>(smem + WS_CTRL_BYTES);
  float* Cs = reinterpret_cast<float*>(smem + WS_CTRL_BYTES + 2 * sizeof(Page));

  if (threadIdx.x == 0) {
    mbar_init(&C->full[0], 128);
    mbar_init(&C->full[1], 128);
    mbar_init(&C->empty[0], 256);
    mbar_init(&C->empty[1], 256);
  }
  __syncthreads();  // ONLY full-block sync: before specialization

  if (threadIdx.x < 128) {
    // ---------------- WG0: scheduler / producer ----------------
    if constexpr (SMR == 40) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if constexpr (SMR == 56) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    const int wtid = threadIdx.x;
    int used[2] = {0, 0};
    for (int cc = 0;; ++cc) {
      const int pg = cc & 1;
      if (wtid == 0) C->claim[pg] = atomicAdd(cursor, 1);
      asm volatile("bar.sync 2, 128;" ::: "memory");
      const int i = C->claim[pg];
      if (used[pg] > 0) mbar_wait(&C->empty[pg], (used[pg] - 1) & 1);  // page free
      if (i >= n_instr) {  // quit: arm this page with idx = -1
        if (wtid == 0) C->slot[pg].idx = -1;
        mbar_arrive(&C->full[pg]);
        break;
      }
      const bf16* Ap = reinterpret_cast<const bf16*>(Aptrs[i]);
      const bf16* Bp = reinterpret_cast<const bf16*>(Bptrs[i]);
      const int md = modes[i];
      if (prefetch_b) issue_B(pages[pg], Bp, md, wtid);  // B needs no dependency
      if (i > 0)
        while (ld_acquire_gpu(done + (i - 1) * 32) == 0) __nanosleep(32);
      if (stamps && wtid == 0) stamps[2 * i] = ws_globaltimer();
      if (!prefetch_b) issue_B(pages[pg], Bp, md, wtid);
      issue_A(pages[pg], Ap, md, wtid);
      if (handoff && wtid != 0) {
        // hw-triggered arrival: fires when THIS thread's prior cp.asyncs complete —
        // no wait_prior stall. Thread 0 keeps the software path so its slot write is
        // release-ordered before the barrier completes (127 hw + 1 sw = 128 arrivals).
        const uint32_t fa = (uint32_t)__cvta_generic_to_shared(&C->full[pg]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(fa));
        __pipeline_commit();  // keep pipeline group bookkeeping consistent
      } else {
        __pipeline_commit();
        __pipeline_wait_prior(0);
        if (wtid == 0) {
          C->slot[pg].cptr = Cptrs[i];
          C->slot[pg].idx = i;
          C->slot[pg].mode = md;
        }
        mbar_arrive(&C->full[pg]);  // consumers go once all 128 arrivals land
      }
      used[pg]++;
      if (done_by == 0) {  // completion on the producer, off the consumer path
        mbar_wait(&C->empty[pg], (used[pg] - 1) & 1);
        if (wtid == 0) {
          if (stamps) stamps[2 * i + 1] = ws_globaltimer();
          st_release_gpu(done + i * 32, 1);
        }
      }
    }
  } else {
    // ---------------- WG1 + WG2: consumers ----------------
    if constexpr (SMR != 0) asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    const int cw = threadIdx.x - 128;  // 0..255
    const int wgc = cw >> 7;           // row half
    const int wtid = cw & 127;
    for (int c = 0;; ++c) {
      const int pg = c & 1, ph = (c >> 1) & 1;
      mbar_wait(&C->full[pg], ph);
      const int i = C->slot[pg].idx;
      if (i < 0) break;
      const int md = C->slot[pg].mode;
      bf16* Cg = reinterpret_cast<bf16*>(C->slot[pg].cptr);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      Page& P = pages[pg];
      uint64_t da[4], db[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        da[s] = (md == 2) ? ws_desc_mn(P.A[wgc][s]) : ws_desc(P.A[wgc][s]);
        db[s] = (md == 1) ? ws_desc_mn(P.B[s]) : ws_desc(P.B[s]);
      }
      float d[32];
#pragma unroll
      for (int z = 0; z < 32; ++z) d[z] = 0.0f;
      if (md == 0)
        ws_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::K>>(da, db, d);
      else if (md == 1)
        ws_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::K, SG::Major::MN>>(da, db, d);
      else
        ws_ktile<SG::MMA_64x64x16_F32BF16BF16_SS<SG::Major::MN, SG::Major::K>>(da, db, d);
      // stage accumulators (PTX m64n64 f32 mapping, ops.cuh recipe)
      {
        const int w = wtid >> 5, l = wtid & 31;
        const int r = wgc * 64 + w * 16 + (l >> 2);
        const int cb = (l & 3) << 1;
#pragma unroll
        for (int n8 = 0; n8 < 8; ++n8)
#pragma unroll
          for (int ii = 0; ii < 2; ++ii)
#pragma unroll
            for (int j = 0; j < 2; ++j)
              Cs[(r + 8 * ii) * WS_LDC + n8 * 8 + cb + j] = d[n8 * 4 + ii * 2 + j];
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (md != 1) {  // normal store: C[128,64] bf16, 1024 uint4 groups
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int gid = cw + g * 256;
          const int m = gid >> 3, c8 = (gid & 7) << 3;
          uint4 out;
          bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
          for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(Cs[m * WS_LDC + c8 + e]);
          *reinterpret_cast<uint4*>(Cg + m * 64 + c8) = out;
        }
      } else {  // NN: store C transposed ([64,128] M-contiguous) for the next TN link
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int gid = cw + g * 256;
          const int n = gid >> 4, m8 = (gid & 15) << 3;
          uint4 out;
          bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
          for (int e = 0; e < 8; ++e) oe[e] = __float2bfloat16(Cs[(m8 + e) * WS_LDC + n]);
          *reinterpret_cast<uint4*>(Cg + n * 128 + m8) = out;
        }
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");  // Cs + stores done block-wide
      if (done_by == 1 && cw == 0) {
        __threadfence();  // cumulative: promotes all consumers' stores to gpu scope
        if (stamps) stamps[2 * i + 1] = ws_globaltimer();
        st_release_gpu(done + i * 32, 1);
      }
      mbar_arrive(&C->empty[pg]);
    }
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void ws_run(torch::Tensor Aptrs, torch::Tensor Bptrs, torch::Tensor Cptrs,
            torch::Tensor modes, torch::Tensor done, torch::Tensor cursor,
            c10::optional<torch::Tensor> stamps, int64_t n, int64_t nblocks,
            int64_t prefetch, int64_t done_by, int64_t smr, int64_t handoff) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute((void*)ws_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         WS_SMEM_REQ);
    cudaFuncSetAttribute((void*)ws_kernel<40>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         WS_SMEM_REQ);
    cudaFuncSetAttribute((void*)ws_kernel<56>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         WS_SMEM_REQ);
    inited = true;
  }
  long long* st =
      stamps.has_value() ? reinterpret_cast<long long*>(stamps->data_ptr<int64_t>()) : nullptr;
  const unsigned long long* ap =
      reinterpret_cast<const unsigned long long*>(Aptrs.data_ptr<int64_t>());
  const unsigned long long* bp =
      reinterpret_cast<const unsigned long long*>(Bptrs.data_ptr<int64_t>());
  const unsigned long long* cp =
      reinterpret_cast<const unsigned long long*>(Cptrs.data_ptr<int64_t>());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smr == 40)
    ws_kernel<40><<<(int)nblocks, 384, WS_SMEM_REQ, stream>>>(
        ap, bp, cp, modes.data_ptr<int>(), (int)n, done.data_ptr<int>(),
        cursor.data_ptr<int>(), st, (int)prefetch, (int)done_by, (int)handoff);
  else if (smr == 56)
    ws_kernel<56><<<(int)nblocks, 384, WS_SMEM_REQ, stream>>>(
        ap, bp, cp, modes.data_ptr<int>(), (int)n, done.data_ptr<int>(),
        cursor.data_ptr<int>(), st, (int)prefetch, (int)done_by, (int)handoff);
  else
    ws_kernel<0><<<(int)nblocks, 384, WS_SMEM_REQ, stream>>>(
        ap, bp, cp, modes.data_ptr<int>(), (int)n, done.data_ptr<int>(),
        cursor.data_ptr<int>(), st, (int)prefetch, (int)done_by, (int)handoff);
  C10_CUDA_CHECK(cudaGetLastError());
}

int64_t ws_smem_used() { return WS_SMEM_USED; }
"""

cpp_src = """
void ws_run(torch::Tensor Aptrs, torch::Tensor Bptrs, torch::Tensor Cptrs,
            torch::Tensor modes, torch::Tensor done, torch::Tensor cursor,
            c10::optional<torch::Tensor> stamps, int64_t n, int64_t nblocks,
            int64_t prefetch, int64_t done_by, int64_t smr, int64_t handoff);
int64_t ws_smem_used();
"""


def build():
    return load_inline(
        name="xorl_ws_probe",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["ws_run", "ws_smem_used"],
        # explicit gencode: CUDA 13's -arch=sm_90a ALSO embeds a compute_90 PTX pass,
        # where unguarded setmaxnreg asm fails ptxas (cute guards its wgmma; we can't).
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-I{CUTE_INC}",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=False,
    )


class Chain:
    """A serial GEMM chain program. kind: 'nt' (all K-major) or 'alt' (NN/TN)."""

    def __init__(self, n, kind="nt", seed=0):
        self.n, self.kind = n, kind
        g = torch.Generator(device="cuda").manual_seed(seed)
        # orthogonal B: chain neither explodes nor vanishes over 256 links
        Bm = torch.linalg.qr(torch.randn(64, 64, generator=g, device="cuda"))[0].contiguous()
        self.Bmat = Bm.to(torch.bfloat16)  # math is always C_i = C_{i-1} @ Bmat^T
        self.a0 = torch.randn(128, 64, generator=g, device="cuda").to(torch.bfloat16)
        self.cs = [torch.empty(128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
        if kind == "nt":
            self.modes = [0] * n
            self.bufB = {0: self.Bmat}  # stored [N,K]
        else:  # alternate NN (even, writes C^T) / TN (odd, reads MN-major A)
            self.modes = [1 if i % 2 == 0 else 2 for i in range(n)]
            self.bufB = {
                1: self.Bmat.t().contiguous(),  # NN: B stored [K,N] = Bmat^T
                2: self.Bmat,
            }  # TN: B stored [N,K]
        aptr = [self.a0.data_ptr()] + [c.data_ptr() for c in self.cs[:-1]]
        self.Aptrs = torch.tensor(aptr, dtype=torch.int64, device="cuda")
        self.Bptrs = torch.tensor([self.bufB[m].data_ptr() for m in self.modes], dtype=torch.int64, device="cuda")
        self.Cptrs = torch.tensor([c.data_ptr() for c in self.cs], dtype=torch.int64, device="cuda")
        self.modes_t = torch.tensor(self.modes, dtype=torch.int32, device="cuda")
        self.done = torch.zeros(n * 32, dtype=torch.int32, device="cuda")
        self.cursor = torch.zeros(1, dtype=torch.int32, device="cuda")

    def stored(self, i):
        """C_i in math layout [128, 64] regardless of storage major."""
        c = self.cs[i]
        if self.modes[i] == 1:  # stored transposed [64,128]
            return c.view(64, 128).t()
        return c

    def run(self, ext, nblocks=132, prefetch=1, done_by=0, smr=0, handoff=0, stamps=None):
        self.done.zero_()
        self.cursor.zero_()
        ext.ws_run(
            self.Aptrs,
            self.Bptrs,
            self.Cptrs,
            self.modes_t,
            self.done,
            self.cursor,
            stamps,
            self.n,
            nblocks,
            prefetch,
            done_by,
            smr,
            handoff,
        )

    def parity(self, ext, cfg, label=""):
        for c in self.cs:
            c.view(-1).fill_(float("nan"))
        self.run(ext, **cfg)
        torch.cuda.synchronize()
        Bf = self.Bmat.float()
        # one-step: kernel C_{i-1} -> one torch hop -> vs kernel C_i (no chain blowup)
        worst = 0.0
        for i in range(self.n):
            prev = self.a0 if i == 0 else self.stored(i - 1)
            ref = (prev.float() @ Bf.T).to(torch.bfloat16)
            err = (self.stored(i).float() - ref.float()).abs().max().item()
            worst = max(worst, err)
        # chain-end drift vs bf16-emulated torch chain (informational)
        c = self.a0
        for i in range(self.n):
            c = (c.float() @ Bf.T).to(torch.bfloat16)
        drift = (self.stored(self.n - 1).float() - c.float()).abs().max().item()
        ok = worst < 0.05
        print(
            f"  parity[{label}] one-step max err {worst:.4e} ({'OK' if ok else 'FAIL'}), chain-end drift {drift:.4e}",
            flush=True,
        )
        return ok

    def bench(self, ext, cfg, iters=20, warmup=5):
        for _ in range(warmup):
            self.run(ext, **cfg)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            self.run(ext, **cfg)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        med = statistics.median(ts)
        return med * 1e3 / (self.n - 1)  # us/hop

    def gap_span(self, ext, cfg):
        stamps = torch.zeros(2 * self.n, dtype=torch.int64, device="cuda")
        self.run(ext, stamps=stamps, **cfg)
        torch.cuda.synchronize()
        clk = stamps.cpu()
        starts, ends = clk[0::2], clk[1::2]
        gaps = (starts[1:] - ends[:-1]).float()  # subtract in int64 FIRST (fp32 eats ns)
        spans = (ends - starts).float()
        return gaps.median().item() / 1e3, spans.median().item() / 1e3


CFGS = [
    ("A serial-load, prod-done", dict(prefetch=0, done_by=0, smr=0)),
    ("B +B-prefetch", dict(prefetch=1, done_by=0, smr=0)),
    ("C +consumer-done", dict(prefetch=1, done_by=1, smr=0)),
    ("D smr40/224, cons-done", dict(prefetch=1, done_by=1, smr=40)),
    ("D' smr40/224, prod-done", dict(prefetch=1, done_by=0, smr=40)),
    ("E smr56/224, prod-done", dict(prefetch=1, done_by=0, smr=56)),
    ("E' smr56/224, cons-done", dict(prefetch=1, done_by=1, smr=56)),
    ("F +async-handoff", dict(prefetch=1, done_by=0, smr=0, handoff=1)),
    ("F' handoff+smr56", dict(prefetch=1, done_by=0, smr=56, handoff=1)),
]


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    ext = build()
    print(f"built. smem used = {ext.ws_smem_used()} B (requested {120 * 1024})", flush=True)

    if stage == "debug":
        ch = Chain(8, "nt")
        for name, cfg in CFGS:
            print(f"debug n=8 nt cfg {name}", flush=True)
            ch.parity(ext, dict(cfg, nblocks=132), label=name)
        ca = Chain(8, "alt")
        for name, cfg in CFGS[:1] + CFGS[3:4]:
            print(f"debug n=8 alt cfg {name}", flush=True)
            ca.parity(ext, dict(cfg, nblocks=132), label=name)
        print("DEBUG DONE", flush=True)
        return

    n = 256
    ch = Chain(n, "nt")
    print(f"\n== NT chain n={n}, 132 blocks ==", flush=True)
    results = []
    for name, cfg in CFGS:
        full = dict(cfg, nblocks=132)
        ok = ch.parity(ext, full, label=name)
        if not ok:
            print(f"  cfg {name}: PARITY FAIL -> timing not counted", flush=True)
            results.append((name, cfg, None, None, None))
            continue
        hop = ch.bench(ext, full)
        gap, span = ch.gap_span(ext, full)
        print(f"  cfg {name:28s}: {hop:6.2f} us/hop  (median gap {gap:5.2f} + span {span:5.2f})", flush=True)
        results.append((name, cfg, hop, gap, span))

    good = [r for r in results if r[2] is not None]
    best = min(good, key=lambda r: r[2])
    print(f"\nbest: {best[0]} at {best[2]:.2f} us/hop", flush=True)

    print(f"\n== alt NN/TN chain n={n}, 132 blocks ==", flush=True)
    ca = Chain(n, "alt")
    for name, cfg in [r[:2] for r in results if r[2] is not None]:
        full = dict(cfg, nblocks=132)
        ok = ca.parity(ext, full, label=name)
        if ok:
            hop = ca.bench(ext, full)
            gap, span = ca.gap_span(ext, full)
            print(f"  cfg {name:28s}: {hop:6.2f} us/hop  (gap {gap:5.2f} + span {span:5.2f})", flush=True)

    print("\n== informational: single-block chain (same-SM hops), best cfg ==", flush=True)
    for nb in (1, 132):
        hop = ch.bench(ext, dict(best[1], nblocks=nb))
        print(f"  nblocks={nb:3d}: {hop:6.2f} us/hop", flush=True)

    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
