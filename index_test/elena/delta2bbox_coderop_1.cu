
#include "elena_int.h"
extern "C" __global__ void coderop_1(float* __restrict__ Placeholder_4, float* __restrict__ Placeholder_5, float* __restrict__ Expand_0, float* __restrict__ Binary_6) {
    const uint64_t iter27_iter28_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter28 = ((iter27_iter28_fused % 1) + 0);
    const uint64_t iter27 = ((iter27_iter28_fused / 1) + 0);
    if ((iter27 < (0 + 3000))) {
        if ((iter28 < (0 + 1))) {
            Expand_0[((iter28 - 0) + (1 * (iter27 - 0)))] = (Placeholder_4[(((1 == 1) ? 0 : iter28) + (1 * ((3000 == 1) ? 0 : iter27)))] + 0);
        }
    }
    const uint64_t iter37 = (((blockIdx.x * 64) + threadIdx.x) + 0);
    if ((iter37 < (0 + max((uint64_t)3000, (uint64_t)3000)))) {
        Binary_6[(iter37 - 0)] = (((Placeholder_5[(1 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter37))))] + 0) + (Placeholder_5[(3 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter37))))] + 0)) * 0.5);
    }
}

