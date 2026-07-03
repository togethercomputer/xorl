// Fused training megakernel — persistent cooperative kernel + in-kernel interpreter.
//
// One kernel executes an entire training fwd+bwd as an instruction stream. Blocks are
// persistent (one per SM); instructions are grouped into dependency-free "waves";
// grid.sync() separates waves; within a wave blocks self-schedule (instr, tile) work
// items. Buffers are referenced through a pointer table so instructions are plain ints.
//
// See NOTES.md for the design; ops.cuh for the device op library.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

#define MK_MAX_ARGS 23
struct Instr {
  int op;
  int tile_off;  // first work-item id of this instr within its wave
  int ntiles;    // work items (block-level tiles) of this instr
  int args[MK_MAX_ARGS];
};
static_assert(sizeof(Instr) == 104, "keep Instr layout in sync with mk.py");

// ---- op enum (mirrored in mk.py; keep in sync) -------------------------------------
enum Op : int {
  OP_NOP = 0,
  OP_FILL_F32 = 1,
  OP_AXPY_F32 = 2,
  OP_GEMM = 3,
  OP_RMSNORM_FWD = 4,
  OP_RMSNORM_BWD = 5,
  OP_SWIGLU_FWD = 6,
  OP_SWIGLU_BWD = 7,
  OP_QKNORM_ROPE_FWD = 8,
  OP_QKNORM_ROPE_BWD = 9,
  OP_EMBED_FWD = 10,
  OP_EMBED_BWD = 11,
  OP_CE_FWD = 12,
  OP_CE_BWD = 13,
  OP_ATTN_FWD = 14,
  OP_ATTN_DPRE = 15,
  OP_ATTN_DKV = 16,
  OP_ATTN_DQ = 17,
};

#include "ops.cuh"
#include "attention.cuh"

// ---- interpreter --------------------------------------------------------------------
__device__ __forceinline__ void dispatch(const Instr& I, int tile, void** bufs,
                                         char* smem) {
  switch (I.op) {
    case OP_NOP:
      break;
    case OP_FILL_F32:
      op_fill_f32(I, tile, bufs);
      break;
    case OP_AXPY_F32:
      op_axpy_f32(I, tile, bufs);
      break;
    case OP_GEMM:
      op_gemm(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_FWD:
      op_rmsnorm_fwd(I, tile, bufs, smem);
      break;
    case OP_RMSNORM_BWD:
      op_rmsnorm_bwd(I, tile, bufs, smem);
      break;
    case OP_SWIGLU_FWD:
      op_swiglu_fwd(I, tile, bufs);
      break;
    case OP_SWIGLU_BWD:
      op_swiglu_bwd(I, tile, bufs);
      break;
    case OP_QKNORM_ROPE_FWD:
      op_qknorm_rope_fwd(I, tile, bufs, smem);
      break;
    case OP_QKNORM_ROPE_BWD:
      op_qknorm_rope_bwd(I, tile, bufs, smem);
      break;
    case OP_EMBED_FWD:
      op_embed_fwd(I, tile, bufs);
      break;
    case OP_EMBED_BWD:
      op_embed_bwd(I, tile, bufs);
      break;
    case OP_CE_FWD:
      op_ce_fwd(I, tile, bufs, smem);
      break;
    case OP_CE_BWD:
      op_ce_bwd(I, tile, bufs);
      break;
    case OP_ATTN_FWD:
      op_attn_fwd(I, tile, bufs, smem);
      break;
    case OP_ATTN_DPRE:
      op_attn_dpre(I, tile, bufs);
      break;
    case OP_ATTN_DKV:
      op_attn_dkv(I, tile, bufs, smem);
      break;
    case OP_ATTN_DQ:
      op_attn_dq(I, tile, bufs, smem);
      break;
    default:
      break;
  }
}

extern "C" __global__ void megakernel(const Instr* __restrict__ instrs,
                                      const int* __restrict__ wave_start,  // [nwaves+1]
                                      const int* __restrict__ wave_tiles,  // [nwaves]
                                      int nwaves, void** bufs,
                                      long long* wave_clk /* nullable [nwaves+1] */) {
  extern __shared__ char smem[];
  cg::grid_group grid = cg::this_grid();
  const int nblocks = gridDim.x;

  if (wave_clk && blockIdx.x == 0 && threadIdx.x == 0) wave_clk[0] = clock64();
  for (int w = 0; w < nwaves; ++w) {
    const int i0 = wave_start[w], i1 = wave_start[w + 1];
    const int total = wave_tiles[w];
    for (int work = blockIdx.x; work < total; work += nblocks) {
      // locate the instruction owning this work item: scan offsets only (2 ints),
      // copy the full Instr struct just for the owner.
      for (int i = i0; i < i1; ++i) {
        if (work < instrs[i].tile_off + instrs[i].ntiles) {
          const Instr I = instrs[i];
          dispatch(I, work - I.tile_off, bufs, smem);
          break;
        }
      }
      __syncthreads();  // smem reuse safety between work items within a wave
    }
    grid.sync();
    if (wave_clk && blockIdx.x == 0 && threadIdx.x == 0) wave_clk[w + 1] = clock64();
  }
}

// ---- host launcher ------------------------------------------------------------------
static int g_nblocks = -1;

void mk_run(torch::Tensor instrs, torch::Tensor wave_start, torch::Tensor wave_tiles,
            torch::Tensor bufs, int64_t smem_bytes,
            c10::optional<torch::Tensor> wave_clk) {
  TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32);
  TORCH_CHECK(wave_start.is_cuda() && wave_tiles.is_cuda() && bufs.is_cuda());
  TORCH_CHECK(bufs.dtype() == torch::kInt64);
  const int nwaves = (int)wave_tiles.numel();

  static bool smem_configured = false;
  if (!smem_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)megakernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes));
    smem_configured = true;
  }
  if (g_nblocks < 0) {
    int dev, sms, per_sm;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)megakernel, 256, (int)smem_bytes));
    TORCH_CHECK(per_sm >= 1, "megakernel does not fit an SM with smem=", smem_bytes);
    g_nblocks = sms * per_sm;
  }

  const Instr* d_instrs = reinterpret_cast<const Instr*>(instrs.data_ptr<int>());
  const int* d_ws = wave_start.data_ptr<int>();
  const int* d_wt = wave_tiles.data_ptr<int>();
  void** d_bufs = reinterpret_cast<void**>(bufs.data_ptr<int64_t>());
  long long* d_clk =
      wave_clk.has_value() ? reinterpret_cast<long long*>(wave_clk->data_ptr<int64_t>()) : nullptr;
  void* args[] = {(void*)&d_instrs, (void*)&d_ws, (void*)&d_wt, (void*)&nwaves,
                  (void*)&d_bufs, (void*)&d_clk};
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)megakernel, dim3(g_nblocks),
                                             dim3(256), args, (size_t)smem_bytes,
                                             stream.stream()));
}

int64_t mk_nblocks() { return g_nblocks; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &mk_run, "run megakernel program");
  m.def("nblocks", &mk_nblocks, "resolved persistent block count");
}
