
#include "elena_int.h"
extern "C" __global__ void coderop_2(float* __restrict__ Placeholder_6, float* __restrict__ Placeholder_7, float* __restrict__ Expand_1, float* __restrict__ Binary_8) {
    const uint64_t iter38_iter39_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter39 = ((iter38_iter39_fused % 1) + 0);
    const uint64_t iter38 = ((iter38_iter39_fused / 1) + 0);
    if ((iter38 < (0 + 3000))) {
        if ((iter39 < (0 + 1))) {
            Expand_1[((iter39 - 0) + (1 * (iter38 - 0)))] = (Placeholder_6[(((1 == 1) ? 0 : iter39) + (1 * ((3000 == 1) ? 0 : iter38)))] + 0);
        }
    }
    const uint64_t iter48 = (((blockIdx.x * 64) + threadIdx.x) + 0);
    if ((iter48 < (0 + max((uint64_t)3000, (uint64_t)3000)))) {
        Binary_8[(iter48 - 0)] = (((Placeholder_7[(2 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter48))))] + 0) - (Placeholder_7[(0 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter48))))] + 0)) + 1);
    }
}

