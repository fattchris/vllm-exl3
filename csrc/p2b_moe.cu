#include <cuda_fp16.h>
#include <torch/extension.h>

// P2B_CB: expert codebook index. 1 = MCG (0xCBAC1FED), 2 = mul1 (0x83DCD12D).
// Build with -DP2B_CB=2 for mul1 packs. The bool arg is only a TORCH_CHECK.
#ifndef P2B_CB
#define P2B_CB 1
#endif
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cmath>
#include <limits>

#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"
// Multi-K additions (ABI 4): exllamav3's dq_dispatch tile decoder
// (exl3_dq.cuh) is templated on bits for K1..8 and is the route-(b)
// decoder for the K5/K6 values the register decoders in exl3_gemv_ns
// (dq8_regs_{2,3,4}bits) never shipped with. The plugin's
// exl3_fat_gemm.cu already compiles this header through the same
// include path (quant/ is on the include list in setup.py).
#include "exl3_dq.cuh"

namespace cg = cooperative_groups;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Multi-K additions (ABI 4). Everything above is the uniform-K path and stays
// byte-identical to the staged original; the blocks below only add to it.
// ---------------------------------------------------------------------------

// Route-(b) tile function: same contract and calling convention as
// run_gemv_tile above (single-row A broadcast through half2 fragments,
// mma_ab_h tensor-core path, fp16 fold every FOLD slices, sh_red warp
// reduction, one 32-column group writeback), but each 16x16 B tile is
// decoded through exllamav3's dq_dispatch (exl3_dq.cuh), which is templated
// on bits for K1..8, instead of the K2/3/4-only register decoders. Tile
// addressing is the standard EXL3 trellis layout [rows/16, cols/16, 16*K]
// int16: one 16x16 tile is 8*bits uint32 words at
//   B32 + row_tile * (ntiles * TWORDS) + col_tile * TWORDS
// exactly the indexing run_gemv_tile derives for its register loads.
template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemm_tile_dq(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_col] : hzero;
            a23[0] = r0_ok ? A2[a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                const uint32_t* tw = B32
                    + (size_t) (ks0 + i) * slice_stride
                    + (size_t) (group * WNT + t) * TWORDS;
                FragB f0, f1;
                dq_dispatch<bits, cb>(tw, lane << 3, f0, f1);
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

// Runtime-K dispatcher: each work item reads its expert's K from an int8
// table and switches over the compiled tile instantiations. The
// instantiation count is the number of distinct K values per projection
// (not the K cross-product): K2/3/4 use the register decoders, K5/6 the
// dq_dispatch tile function above. The switch operand is block-uniform
// (src = ids[e] is uniform per work item), so no warp divergence is added.
template <int CB>
__device__ __forceinline__ void run_gemv_tile_k(
    int bits,
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    switch (bits) {
        case 2: run_gemv_tile<2, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 3: run_gemv_tile<3, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 4: run_gemv_tile<4, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 5: run_gemm_tile_dq<5, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 6: run_gemm_tile_dq<6, CB, 0>(B32, A2, C, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        default: {
            // Unreachable: Python validates the table values (2..6) when the
            // tables are built in finalize. Zero the tile deterministically
            // so a bad table can never propagate garbage into the reduction.
            for (int idx = threadIdx.x; idx < 32; idx += 512)
                C[group * 32 + idx] = __float2half_rn(0.0f);
            break;
        }
    }
}

// _GROUPED: expert groups of live routing slots, rebuilt by stage 0 every launch.
#define P2B_MAX_GROUPS 1024
#define P2B_GROUP_ROWS 16
__device__ int p2b_g_ngroups;
__device__ int p2b_g_expert[P2B_MAX_GROUPS];
__device__ int p2b_g_count[P2B_MAX_GROUPS];
__device__ int p2b_g_slots[P2B_MAX_GROUPS][P2B_GROUP_ROWS];

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile_m(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2base, int a_stride,
    const int* __restrict__ rows, int nrows,
    half* __restrict__ Cbase, int c_stride,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[16][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
        // _GROUPED: A-fragment row g = lane>>2 ([0] regs) and g+8 ([1] regs).
    const int g_row = lane >> 2;
    const int slot_a = g_row < nrows ? rows[g_row] : -1;
    const int slot_b = g_row + 8 < nrows ? rows[g_row + 8] : -1;
    const half2* __restrict__ pa = A2base + (size_t) (slot_a < 0 ? 0 : slot_a) * a_stride;
    const half2* __restrict__ pb = A2base + (size_t) (slot_b < 0 ? 0 : slot_b) * a_stride;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};
    float2 acc1[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = slot_a >= 0 ? pa[a_col] : hzero;
            a23[0] = slot_a >= 0 ? pa[a_col + 4] : hzero;
            a01[1] = slot_b >= 0 ? pb[a_col] : hzero;
            a23[1] = slot_b >= 0 ? pb[a_col + 4] : hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                        acc1[t][f].x += __low2float(ch[t][f][1]);
                        acc1[t][f].y += __high2float(ch[t][f][1]);
                        ch[t][f][1] = hzero;
                    }
            }
        }
    }

    // Warp reduction, per A row (rows g and g+8 of every lane).
    #pragma unroll
    for (int t = 0; t < WNT; ++t) {
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + (lane & 3) * 2;
            if (g_row < nrows) {
                sh_red[warp][g_row][col + 0] = acc0[t][f].x;
                sh_red[warp][g_row][col + 1] = acc0[t][f].y;
            }
            if (g_row + 8 < nrows) {
                sh_red[warp][g_row + 8][col + 0] = acc1[t][f].x;
                sh_red[warp][g_row + 8][col + 1] = acc1[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < nrows * COLS; idx += THREADS) {
        const int r = idx / COLS;
        const int c = idx % COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][r][c];
        Cbase[(size_t) rows[r] * c_stride + group * COLS + c] = __float2half_rn(sum);
    }
    __syncthreads();
}

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemm_tile_dq_m(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2base, int a_stride,
    const int* __restrict__ rows, int nrows,
    half* __restrict__ Cbase, int c_stride,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[16][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

        // _GROUPED: A-fragment row g = lane>>2 ([0] regs) and g+8 ([1] regs).
    const int g_row = lane >> 2;
    const int slot_a = g_row < nrows ? rows[g_row] : -1;
    const int slot_b = g_row + 8 < nrows ? rows[g_row + 8] : -1;
    const half2* __restrict__ pa = A2base + (size_t) (slot_a < 0 ? 0 : slot_a) * a_stride;
    const half2* __restrict__ pb = A2base + (size_t) (slot_b < 0 ? 0 : slot_b) * a_stride;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};
    float2 acc1[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = slot_a >= 0 ? pa[a_col] : hzero;
            a23[0] = slot_a >= 0 ? pa[a_col + 4] : hzero;
            a01[1] = slot_b >= 0 ? pb[a_col] : hzero;
            a23[1] = slot_b >= 0 ? pb[a_col + 4] : hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                const uint32_t* tw = B32
                    + (size_t) (ks0 + i) * slice_stride
                    + (size_t) (group * WNT + t) * TWORDS;
                FragB f0, f1;
                dq_dispatch<bits, cb>(tw, lane << 3, f0, f1);
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                        acc1[t][f].x += __low2float(ch[t][f][1]);
                        acc1[t][f].y += __high2float(ch[t][f][1]);
                        ch[t][f][1] = hzero;
                    }
            }
        }
    }

    // Warp reduction, per A row (rows g and g+8 of every lane).
    #pragma unroll
    for (int t = 0; t < WNT; ++t) {
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + (lane & 3) * 2;
            if (g_row < nrows) {
                sh_red[warp][g_row][col + 0] = acc0[t][f].x;
                sh_red[warp][g_row][col + 1] = acc0[t][f].y;
            }
            if (g_row + 8 < nrows) {
                sh_red[warp][g_row + 8][col + 0] = acc1[t][f].x;
                sh_red[warp][g_row + 8][col + 1] = acc1[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < nrows * COLS; idx += THREADS) {
        const int r = idx / COLS;
        const int c = idx % COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][r][c];
        Cbase[(size_t) rows[r] * c_stride + group * COLS + c] = __float2half_rn(sum);
    }
    __syncthreads();
}

template <int CB>
__device__ __forceinline__ void run_gemv_tile_k_m(
    int bits,
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2base, int a_stride,
    const int* __restrict__ rows, int nrows,
    half* __restrict__ Cbase, int c_stride,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[16][32])
{
    switch (bits) {
        case 2: run_gemv_tile_m<2, CB, 0>(B32, A2base, a_stride, rows, nrows, Cbase, c_stride, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 3: run_gemv_tile_m<3, CB, 0>(B32, A2base, a_stride, rows, nrows, Cbase, c_stride, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 4: run_gemv_tile_m<4, CB, 0>(B32, A2base, a_stride, rows, nrows, Cbase, c_stride, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 5: run_gemm_tile_dq_m<5, CB, 0>(B32, A2base, a_stride, rows, nrows, Cbase, c_stride, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        case 6: run_gemm_tile_dq_m<6, CB, 0>(B32, A2base, a_stride, rows, nrows, Cbase, c_stride, kslices, size_k, group, ntiles, warp, lane, sh_red); break;
        default: {
            // Unreachable: Python validates the table values (2..6) when the
            // tables are built in finalize. Zero the tile deterministically
            // so a bad table can never propagate garbage into the reduction.
            for (int idx = threadIdx.x; idx < nrows * 32; idx += 512)
                Cbase[(size_t) rows[idx / 32] * c_stride + group * 32 + idx % 32] = __float2half_rn(0.0f);
            break;
        }
    }
}


template <int BITS>
__global__ __launch_bounds__(512)
void p2b_moe_batched_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;

            had_hf_r_128_inner<true, false>(x + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;

            run_gemv_tile<BITS, P2B_CB, 0>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain activation.
        int total_elements = experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;

            run_gemv_tile<BITS, P2B_CB, 0>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum
        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            float w = __half2float(rw[e]);
            atomicAdd(accum + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

// Mixed-K cooperative MoE kernel (ABI 4): ONE launch for a routing list whose
// experts carry per-expert K (SAGE-allocated packs). The phase structure is
// identical to p2b_moe_batched_kernel above; the differences are:
//   * per-expert K comes from int8 tables kg_tab/ku_tab/kd_tab indexed by the
//     block-uniform src = ids[e], and phases 2/4 dispatch through
//     run_gemv_tile_k instead of the compile-time BITS template;
//   * n_local bounds the pointer tables: routing slots with
//     src >= n_local (the EP sentinel produced by map_topk_to_local) are
//     skipped in every phase, so a non-local slot neither streams its
//     expert's weights nor lets uninitialized fp16 scratch (possibly NaN)
//     reach the fp32 accumulation. The guard in the weighted reduction is
//     load-bearing: an uninitialized half can be NaN and 0 * NaN = NaN would
//     poison the output even with a zeroed routing weight.
__global__ __launch_bounds__(512)
void p2b_moe_mixedk_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    int n_local,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;

            had_hf_r_128_inner<true, false>(x + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;
            const int kb = (int) (is_up ? ku_tab[src] : kg_tab[src]);

            run_gemv_tile_k<P2B_CB>(kb, B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain activation.
        // Sentinel slots are skipped so uninitialized gate/up rows (possibly
        // NaN/Inf) never enter the activation pipeline at all.
        int total_elements = experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e3 = j / inter;
            int src3 = ids[e3];
            if (src3 < 0 || src3 >= n_local) continue;
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;
            const int kb = (int) kd_tab[src];

            run_gemv_tile_k<P2B_CB>(kb, B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum. The sentinel guard is load-bearing:
        // skipped experts leave `down` uninitialized and an uninitialized
        // half can be NaN, so 0 * NaN = NaN would poison accum even though
        // the routing weight is zero.
        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            int src5 = ids[e];
            if (src5 < 0 || src5 >= n_local) continue;
            float w = __half2float(rw[e]);
            atomicAdd(accum + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

template <int BITS>
static void launch_moe_batched(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_batched_kernel<BITS>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_moe_mixedk(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int n_local,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    // Re-query occupancy for the (larger) mixed-K kernel; never reuse the
    // uniform-K template's resident count.
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_mixedk_kernel;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());
    const int8_t* kgp = kg_tab.data_ptr<int8_t>();
    const int8_t* kup = ku_tab.data_ptr<int8_t>();
    const int8_t* kdp = kd_tab.data_ptr<int8_t>();

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&kgp, (void*)&kup, (void*)&kdp, (void*)&n_local,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Host entry for the mixed-K cooperative MoE kernel (ABI 4): one launch for a
// per-expert-K routing list. Same contract as p2b_fused_moe_cuda above
// (single input row, 128-aligned geometry, pointer tables validated by the
// Python caller) except:
//   * the K triple is per expert, carried by int8 device tables
//     kg_tab/ku_tab/kd_tab with one entry per pointer-table slot
//     (K = trellis.shape[-1] / 16 per projection, built by Python finalize);
//   * ids may carry the n_local sentinel (== pointer-table length) for
//     non-local routed slots; the kernel skips those slots in every phase,
//     so their routing weights are irrelevant (Python zeroes them for
//     cleanliness only).
// Table VALUES are validated on the host at load time (Python finalize);
// this entry point must not add a device->host read on the decode path.
at::Tensor p2b_fused_moe_mk_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int64_t n_local, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "mixed-K fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "mixed-K fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "mixed-K fused MoE requires exactly one input row");
    TORCH_CHECK(out.sizes() == x.sizes(), "mixed-K fused MoE output shape must match input");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0,
                "mixed-K fused MoE hidden width must be a positive multiple of 128");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "mixed-K fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max() &&
                intermediate_size <= std::numeric_limits<int>::max(),
                "mixed-K fused MoE dimensions exceed int32 kernel indexing");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "mixed-K fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(mcg == (P2B_CB == 1), "codebook mismatch: this kernel was built for cb=", P2B_CB, " but the pack reports ", mcg ? "MCG" : "mul1");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt && ids.numel() > 0,
                "mixed-K fused MoE expert indices must be a nonempty int32 routing vector");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.numel() == ids.numel(),
                "mixed-K fused MoE requires one fp16 routing weight per expert index");
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv,
                                   &kg_tab, &ku_tab, &kd_tab};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "mixed-K fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "mixed-K fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    for (const auto* tab : {&kg_tab, &ku_tab, &kd_tab}) {
        TORCH_CHECK(tab->is_cuda() && tab->dim() == 1 && tab->is_contiguous() &&
                    tab->scalar_type() == at::kChar && tab->numel() == gt.numel(),
                    "mixed-K fused MoE requires int8 CUDA K tables with one entry per pointer-table slot");
    }
    TORCH_CHECK(n_local >= 1 && n_local <= static_cast<int64_t>(gt.numel()),
                "mixed-K fused MoE n_local must bound the pointer tables (1 <= n_local <= table length)");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, m, inter}, x.options());
    auto up = at::empty({e, m, inter}, x.options());
    auto down = at::empty({e, m, hidden}, x.options());
    auto had_gate = at::empty({e, m, hidden}, x.options());
    auto had_up = at::empty({e, m, hidden}, x.options());
    auto had_down = at::empty({e, m, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    launch_moe_mixedk(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw,
                      kg_tab, ku_tab, kd_tab, static_cast<int>(n_local),
                      out, gate, up, down, had_gate, had_up, had_down,
                      accum, e, m, hidden, inter, swiglu_limit);

    return out;
}


// ---------------------------------------------------------------------------
// Padded fixed-shape additions (ABI 4, capability flag P2B_MOE_PADDED).
// Everything above — the uniform-K path AND the shipped mixed-K path — stays
// byte-identical to the staged PR #31 sources; the blocks below only add.
// ---------------------------------------------------------------------------

// Padded fixed-shape cooperative MoE kernel: ONE launch for the whole
// [MAX_T, MAX_K] padded routing grid, whose live row count is a DEVICE fact
// (n_valid_dev[0]) so the decode apply path never performs a device->host
// read. Phase structure is identical to p2b_moe_mixedk_kernel above; the
// differences are:
//   * the routing list is the flattened padded grid ids[MAX_T*MAX_K]; slot e
//     addresses token tok = e / max_k, and every per-slot phase loop carries
//     a token guard (tok >= n_valid_dev[0]) next to the sentinel guard — so
//     padded rows are skipped REGARDLESS of their content and stale ids from
//     a previous longer step can never reach pointer tables or the
//     accumulation;
//   * phase 1 reads the token's own input row (x + tok * hidden + w * 128)
//     and the weighted reduction is DETERMINISTIC: one block per output
//     token loops that token's K experts sequentially in expert-index
//     order, accumulates the fp32 products w * down in registers, and
//     writes each accum element exactly once — there is NO atomicAdd in the
//     token-sum path (float atomicAdd is ordered but its serialization
//     order is scheduler-dependent, which made the cross-expert sum round
//     differently run to run at T >= 6 and flipped greedy verify tokens);
//   * the write-back covers all MAX_T rows: padded rows emit zeros (their
//     block writes sum = 0.0f because the token guard fails).
__global__ __launch_bounds__(512)
void p2b_moe_padded_stage0(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.
    for (int j = tid; j < max_t * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    {
        int warps_per_exp = hidden / 128;
        int total_warps = live * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;

            had_hf_r_128_inner<true, false>(x + (size_t) tok * hidden + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + (size_t) tok * hidden + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }

    }

    // _GROUPED: block 0 groups the live, valid slots by expert (<=16 per group),
    // in slot order, via a per-expert "open group" table in shared memory.
    if (blockIdx.x == 0) {
        __shared__ int emap[1024];
        const bool use_map = n_local <= 1024;
        if (use_map)
            for (int q = threadIdx.x; q < n_local; q += blockDim.x) emap[q] = -1;
        __syncthreads();
        if (threadIdx.x == 0) {
            int ng = 0;
            const int nv = n_valid_dev[0];
            for (int e = 0; e < live; ++e) {
                const int src = ids[e];
                if (src < 0 || src >= n_local || e / max_k >= nv) continue;
                int g = -1;
                if (use_map) {
                    g = emap[src];
                } else {
                    for (int q = ng - 1; q >= 0; --q)
                        if (p2b_g_expert[q] == src) { g = q; break; }
                }
                if (g < 0 || p2b_g_count[g] == P2B_GROUP_ROWS) {
                    if (ng == P2B_MAX_GROUPS) continue;
                    g = ng++;
                    p2b_g_expert[g] = src;
                    p2b_g_count[g] = 0;
                    if (use_map) emap[src] = g;
                }
                p2b_g_slots[g][p2b_g_count[g]++] = e;
            }
            p2b_g_ngroups = ng;
        }
    }
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage1(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Phase 2: Batched Gate & Up GEMV across all routing-grid slots
    {
        int total_work = 2 * live * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;
            const int kb = (int) (is_up ? ku_tab[src] : kg_tab[src]);

            run_gemv_tile_k<P2B_CB>(kb, B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }

    }
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage2(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = live * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }

    }
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage3(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Phase 3: SwiGLU activation + Down input Hadamard across all slots
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain
        // activation. Sentinel AND padded slots are skipped so uninitialized
        // gate/up rows (possibly NaN/Inf) never enter the pipeline.
        int total_elements = live * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e3 = j / inter;
            int tok3 = e3 / max_k;
            int src3 = ids[e3];
            if (src3 < 0 || src3 >= n_local) continue;
            if (tok3 >= n_valid_dev[0]) continue;
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }

}
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage4(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = live * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }

}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage5(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Phase 4: Batched Down GEMV across all routing-grid slots
    {
        int total_work = live * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;
            const int kb = (int) kd_tab[src];

            run_gemv_tile_k<P2B_CB>(kb, B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }

    }
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage6(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Down output Hadamard, then DETERMINISTIC per-token weighted reduction
    {
        int warps_per_exp = hidden / 128;
        int total_warps = live * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int tok = e / max_k;
            int src = ids[e];
            if (src < 0 || src >= n_local) continue;
            if (tok >= n_valid_dev[0]) continue;
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }

}
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage7(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
        // Deterministic weighted reduction (verify-blocking bug fix): ONE
        // BLOCK PER OUTPUT TOKEN. Each block walks its token's max_k slots
        // sequentially in expert-index order (e = tok * max_k + k, k
        // ascending), accumulates the fp32 products w * down in a register,
        // and writes each accum element exactly once — no atomics in the
        // token-sum path. The atomics this replaces were ordered but their
        // serialization order was scheduler-dependent, so the cross-expert
        // sum rounded differently run to run at T >= 6; a fixed expert order
        // now rounds identically on every run. The per-expert GEMV partial
        // sums above keep their existing deterministic block reductions —
        // only the cross-expert sum order changed. Both guards stay
        // load-bearing (cf. the mixed-K comment above): a skipped slot's
        // `down` row is uninitialized and an uninitialized half can be NaN,
        // so 0 * NaN = NaN would poison the token's accum even with a
        // zeroed routing weight — skipped slots are never read; a padded
        // token's row is simply written as exact zeros (sum stays 0.0f).
        const int n_valid_red = n_valid_dev[0];
        for (int tok = blockIdx.x; tok < max_t; tok += gridDim.x) {
            const bool live_tok = tok < n_valid_red;
            for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
                float sum = 0.0f;
                if (live_tok) {
                    for (int k = 0; k < max_k; ++k) {
                        const int e = tok * max_k + k;
                        const int src = ids[e];
                        if (src < 0 || src >= n_local) continue;
                        const float w = __half2float(rw[e]);
                        sum += w * __half2float(down[(size_t) e * hidden + col]);
                    }
                }
                accum[(size_t) tok * hidden + col] = sum;
            }
        }

}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage8(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Write back to out: every padded row is exactly zero (its block wrote
    // sum = 0.0f in the deterministic reduction: the token guard failed).
    for (int j = tid; j < max_t * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }

}


__global__ __launch_bounds__(512)
void p2b_moe_padded_stage1_grouped(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Phase 2 (_GROUPED): Gate & Up GEMV, one work item per (expert group, col group).
    {
        __shared__ float sh_red_m[16][16][32];
        const int ng = p2b_g_ngroups;
        int total_work = 2 * ng * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int g = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = p2b_g_expert[g];
            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2b = reinterpret_cast<const half2*>(is_up ? had_up : had_gate);
            half* Cb = is_up ? up : gate;
            const int kb = (int) (is_up ? ku_tab[src] : kg_tab[src]);
            run_gemv_tile_k_m<P2B_CB>(kb, B32, A2b, hidden / 2, p2b_g_slots[g], p2b_g_count[g],
                                      Cb, inter, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red_m);
        }
    }
}

__global__ __launch_bounds__(512)
void p2b_moe_padded_stage5_grouped(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    const int8_t* __restrict__ kg_tab,
    const int8_t* __restrict__ ku_tab,
    const int8_t* __restrict__ kd_tab,
    const int32_t* __restrict__ n_valid_dev,
    int n_local,
    int max_k,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int hidden,
    int inter,
    float swiglu_limit)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // experts == MAX_T * MAX_K (host validates ids == [x.size(0), max_k]).
    const int max_t = experts / max_k;
    // Only the live prefix of the routing grid (n_valid rows) is iterated;
    // slots past it are padding and were previously skipped one by one.
    const int live = min(experts, n_valid_dev[0] * max_k);

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum for every padded row (defensive only): the deterministic
    // reduction below rewrites every row — padded rows as exact zeros — so
    // the write-back emits zeros for them.

    // Phase 1: Input Hadamard for Gate and Up across all routing-grid slots
    
    // Phase 4 (_GROUPED): Down GEMV, one work item per (expert group, col group).
    {
        __shared__ float sh_red_m[16][16][32];
        const int ng = p2b_g_ngroups;
        int total_work = ng * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int g = item / num_groups_down;
            int group = item % num_groups_down;
            int src = p2b_g_expert[g];
            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const int kb = (int) kd_tab[src];
            run_gemv_tile_k_m<P2B_CB>(kb, B32, reinterpret_cast<const half2*>(had_down), inter / 2,
                                      p2b_g_slots[g], p2b_g_count[g], down, hidden,
                                      kslices_down, inter, group, ntiles_down, warp, lane, sh_red_m);
        }
    }
}

static void launch_moe_padded(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    const at::Tensor& n_valid,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int n_local, int max_k,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int hidden, int inter, float swiglu_limit)
{
    // Re-query occupancy for the padded kernel; never reuse the mixed-K or
    // uniform-K resident counts (same rule as launch_moe_mixedk above — a
    // stale occupancy figure for a cooperative launch is a correctness bug).
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_padded_stage0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());
    // The ONLY host-side touch of n_valid: extracting the device pointer.
    // Its value is read exclusively by the kernel's token guard.
    const int32_t* nvp = n_valid.data_ptr<int32_t>();
    const int8_t* kgp = kg_tab.data_ptr<int8_t>();
    const int8_t* kup = ku_tab.data_ptr<int8_t>();
    const int8_t* kdp = kd_tab.data_ptr<int8_t>();

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&kgp, (void*)&kup, (void*)&kdp,
        (void*)&nvp, (void*)&n_local, (void*)&max_k,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    // _CAPTURE_SPLIT (Astra A(i) / Fable captest2): ordinary same-stream launches.
    // The cooperative launch was never recorded by relaxed capture (proven: canary
    // survived replay), so the phases become independent kernels. Capture-legal.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, (void*) p2b_moe_padded_stage0, 512, 0);
    const int grid_split = std::max(1, resident * sms);
    static const bool grouped = [] {
        const char* v = getenv("P2B_GROUPED");
        return !(v && v[0] == '0');
    }();
    void* kernels[] = {
        (void*) p2b_moe_padded_stage0,
        grouped ? (void*) p2b_moe_padded_stage1_grouped : (void*) p2b_moe_padded_stage1,
        (void*) p2b_moe_padded_stage2,
        (void*) p2b_moe_padded_stage3,
        (void*) p2b_moe_padded_stage4,
        grouped ? (void*) p2b_moe_padded_stage5_grouped : (void*) p2b_moe_padded_stage5,
        (void*) p2b_moe_padded_stage6,
        (void*) p2b_moe_padded_stage7,
        (void*) p2b_moe_padded_stage8,
    };
    for (void* kfn : kernels) {
        cuda_check(cudaLaunchKernel(kfn, dim3(grid_split), dim3(512), args, 0, stream));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Host entry for the padded fixed-shape cooperative MoE kernel (ABI 4,
// capability flag P2B_MOE_PADDED): ONE launch for the whole [MAX_T, MAX_K]
// padded routing grid. Same pointer/K-table contract as
// p2b_fused_moe_mk_cuda above, except:
//   * x/out are [MAX_T, hidden] and ids/rw are the flattened [MAX_T, MAX_K]
//     grid (ids must be 2-D with ids.size(0) == x.size(0));
//   * the live row count is a DEVICE fact carried by n_valid (int32[1]).
//     This entry validates its METADATA (CUDA, int32, numel == 1,
//     contiguous) and NEVER reads its value — the host knows only
//     MAX_T = x.size(0) from static shape metadata, and reading the count
//     would put a device->host sync on the decode path. n_valid's data
//     pointer is handed to the kernel via launch_moe_padded.
// Padded rows (tok >= n_valid on the device) are skipped by the kernel's
// token guard in every phase regardless of their ids/weights content, so
// the write-back emits exactly zeros for them; the Python wrapper returns
// the shape-static view out[:T].
at::Tensor p2b_fused_moe_padded_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, const at::Tensor& n_valid,
    const at::Tensor& kg_tab, const at::Tensor& ku_tab, const at::Tensor& kd_tab,
    int64_t n_local, int64_t max_k, bool mcg, int64_t intermediate_size,
    float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "padded fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "padded fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1,
                "padded fused MoE input must be a [MAX_T, hidden] matrix");
    TORCH_CHECK(out.sizes() == x.sizes(), "padded fused MoE output shape must match input");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0,
                "padded fused MoE hidden width must be a positive multiple of 128");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "padded fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max() &&
                intermediate_size <= std::numeric_limits<int>::max(),
                "padded fused MoE dimensions exceed int32 kernel indexing");
    TORCH_CHECK(x.size(0) * max_k <= std::numeric_limits<int>::max() &&
                x.size(0) * x.size(1) <= std::numeric_limits<int>::max(),
                "padded fused MoE grid exceeds int32 kernel indexing");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "padded fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(mcg == (P2B_CB == 1),
                "padded fused MoE codebook mismatch: this kernel was built for cb=", P2B_CB,
                " but the pack reports ", mcg ? "MCG" : "mul1");
    TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == at::kInt &&
                ids.size(0) == x.size(0) && ids.size(1) >= 1,
                "padded fused MoE requires an int32 [MAX_T, MAX_K] routing grid whose rows match x");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.sizes() == ids.sizes(),
                "padded fused MoE requires one fp16 routing weight per routing-grid slot");
    // n_valid: metadata only. Its VALUE is a device fact read by the kernel's
    // token guard; this host entry must never dereference it (no .item call,
    // no host copy, no indexing) — that would be the device->host sync this
    // entry exists to eliminate.
    TORCH_CHECK(n_valid.is_cuda() && n_valid.dim() == 1 && n_valid.numel() == 1 &&
                n_valid.scalar_type() == at::kInt && n_valid.is_contiguous() &&
                n_valid.device() == x.device(),
                "padded fused MoE requires a contiguous int32 CUDA n_valid scalar tensor on x.device()");
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv,
                                   &kg_tab, &ku_tab, &kd_tab};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "padded fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "padded fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    for (const auto* tab : {&kg_tab, &ku_tab, &kd_tab}) {
        TORCH_CHECK(tab->is_cuda() && tab->dim() == 1 && tab->is_contiguous() &&
                    tab->scalar_type() == at::kChar && tab->numel() == gt.numel(),
                    "padded fused MoE requires int8 CUDA K tables with one entry per pointer-table slot");
    }
    TORCH_CHECK(n_local >= 1 && n_local <= static_cast<int64_t>(gt.numel()),
                "padded fused MoE n_local must bound the pointer tables (1 <= n_local <= table length)");
    TORCH_CHECK(max_k == ids.size(1),
                "padded fused MoE max_k must equal the routing grid's column count");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.numel());
    const int max_t = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, 1, inter}, x.options());
    auto up = at::empty({e, 1, inter}, x.options());
    auto down = at::empty({e, 1, hidden}, x.options());
    auto had_gate = at::empty({e, 1, hidden}, x.options());
    auto had_up = at::empty({e, 1, hidden}, x.options());
    auto had_down = at::empty({e, 1, inter}, x.options());
    auto accum = at::zeros({max_t, hidden}, x.options().dtype(at::kFloat));

    launch_moe_padded(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, n_valid,
                      kg_tab, ku_tab, kd_tab, static_cast<int>(n_local),
                      static_cast<int>(max_k),
                      out, gate, up, down, had_gate, had_up, had_down,
                      accum, e, hidden, inter, swiglu_limit);

    return out;
}


at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "fused MoE requires exactly one input row");
    TORCH_CHECK(out.sizes() == x.sizes(), "fused MoE output shape must match input");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE hidden width must be a positive multiple of 128");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max() &&
                intermediate_size <= std::numeric_limits<int>::max(),
                "fused MoE dimensions exceed int32 kernel indexing");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(mcg == (P2B_CB == 1) && kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4),
                "unsupported fused MoE K, or codebook mismatch: this kernel was built for cb=", P2B_CB);
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt && ids.numel() > 0,
                "fused MoE expert indices must be a nonempty int32 routing vector");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.numel() == ids.numel(),
                "fused MoE requires one fp16 routing weight per expert index");
    // Pointer tables describe already-loaded tensors. Their pointee shapes and
    // expert IDs are validated/prepared by the Python caller, without a host sync.
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, m, inter}, x.options());
    auto up = at::empty({e, m, inter}, x.options());
    auto down = at::empty({e, m, hidden}, x.options());
    auto had_gate = at::empty({e, m, hidden}, x.options());
    auto had_up = at::empty({e, m, hidden}, x.options());
    auto had_down = at::empty({e, m, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    if (kg == 2) launch_moe_batched<2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (kg == 3) launch_moe_batched<3>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (kg == 4) launch_moe_batched<4>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);

    return out;
}