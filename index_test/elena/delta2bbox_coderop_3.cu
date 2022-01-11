
#include "elena_int.h"
extern "C" __global__ void coderop_3(float* __restrict__ Placeholder_8, float* __restrict__ Placeholder_9, float* __restrict__ Expand_2, float* __restrict__ Binary_10) {
    const uint64_t iter49_iter50_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter50 = ((iter49_iter50_fused % 1) + 0);
    const uint64_t iter49 = ((iter49_iter50_fused / 1) + 0);
    if ((iter49 < (0 + 3000))) {
        if ((iter50 < (0 + 1))) {
            Expand_2[((iter50 - 0) + (1 * (iter49 - 0)))] = (Placeholder_8[(((1 == 1) ? 0 : iter50) + (1 * ((3000 == 1) ? 0 : iter49)))] + 0);
        }
    }
    const uint64_t iter59 = (((blockIdx.x * 64) + threadIdx.x) + 0);
    if ((iter59 < (0 + max((uint64_t)3000, (uint64_t)3000)))) {
        Binary_10[(iter59 - 0)] = (((Placeholder_9[(3 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter59))))] + 0) - (Placeholder_9[(1 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter59))))] + 0)) + 1);
    }
}

