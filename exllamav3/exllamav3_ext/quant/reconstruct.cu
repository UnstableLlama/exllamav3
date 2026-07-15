#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reconstruct.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"

// Output-dtype conversion at the tile-write stage. The dequant math is
// fp16 throughout; BF16_OUT applies one round-to-nearest fp16->bf16
// conversion per element, bit-identical to reconstructing in half and
// casting with .to(bf16) afterwards. Both types are 2 bytes, so the tile
// layout and the vectorized int4 store are shared (tile holds raw bits).
template <bool BF16_OUT>
__device__ inline uint32_t out_pack(half2 v)
{
    if constexpr (BF16_OUT)
    {
        __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(v));
        return *reinterpret_cast<uint32_t*>(&b);
    }
    else
        return *reinterpret_cast<uint32_t*>(&v);
}

template <int K, int cb, bool BF16_OUT>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    void* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile (raw 2x16-bit
    // element pairs; half2 or bfloat162 bits depending on BF16_OUT)
    __shared__ uint32_t tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = out_pack<BF16_OUT>(m0);
        tile[r1][warp_id][c0] = out_pack<BF16_OUT>(m1);
        tile[r2][warp_id][c0] = out_pack<BF16_OUT>(m2);
        tile[r3][warp_id][c0] = out_pack<BF16_OUT>(m3);
        tile[r0][warp_id][c1] = out_pack<BF16_OUT>(m4);
        tile[r1][warp_id][c1] = out_pack<BF16_OUT>(m5);
        tile[r2][warp_id][c1] = out_pack<BF16_OUT>(m6);
        tile[r3][warp_id][c1] = out_pack<BF16_OUT>(m7);
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

#define __(i, cb, bf) reconstruct_kernel<i, cb, bf>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0, false), __(2, 0, false), __(3, 0, false), __(4, 0, false), __(5, 0, false), __(6, 0, false), __(7, 0, false), __(8, 0, false),
    __(1, 1, false), __(2, 1, false), __(3, 1, false), __(4, 1, false), __(5, 1, false), __(6, 1, false), __(7, 1, false), __(8, 1, false),
    __(1, 2, false), __(2, 2, false), __(3, 2, false), __(4, 2, false), __(5, 2, false), __(6, 2, false), __(7, 2, false), __(8, 2, false),
    __(1, 0, true),  __(2, 0, true),  __(3, 0, true),  __(4, 0, true),  __(5, 0, true),  __(6, 0, true),  __(7, 0, true),  __(8, 0, true),
    __(1, 1, true),  __(2, 1, true),  __(3, 1, true),  __(4, 1, true),  __(5, 1, true),  __(6, 1, true),  __(7, 1, true),  __(8, 1, true),
    __(1, 2, true),  __(2, 2, true),  __(3, 2, true),  __(4, 2, true),  __(5, 2, true),  __(6, 2, true),  __(7, 2, true),  __(8, 2, true)
};
#undef __

/*
Reconstruct encoded+packed tensor
*/
void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);
    bool bf16_out = unpacked.dtype() == at::kBFloat16;
    TORCH_CHECK(bf16_out || unpacked.dtype() == at::kHalf,
        "unpacked is incorrect datatype, must be kHalf or kBFloat16");

    int rows = packed.size(0);
    int packed_cols = packed.size(1);

    if (unpacked.numel() == 0)
        return;

    TORCH_CHECK(unpacked.size(1) % 128 == 0, "unpacked N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");

    int cols = unpacked.size(1) / 16;
    int packed_n_offset = n_offset / 16;

    dim3 blockDim(256);
    dim3 gridDim(cols / 8, rows);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    if (bf16_out) cbi += 24;

    reconstruct_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>
    (
        unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        packed_cols,
        packed_n_offset
    );
    cuda_check(cudaPeekAtLastError());
}

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    TORCH_CHECK_SHAPES(unpacked, 1, packed, 1, 16);
    reconstruct_slice(unpacked, packed, K, mcg, mul1, 0);
}
