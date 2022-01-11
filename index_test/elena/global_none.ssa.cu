#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Binary_1) {
    const uint64_t iter4_iter5_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter5 = ((iter4_iter5_fused % max((uint64_t)4, (uint64_t)4)) + 0);
    const uint64_t iter4 = ((iter4_iter5_fused / max((uint64_t)4, (uint64_t)4)) + 0);
    if ((iter4 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)256)))) {
        if ((iter5 < (0 + max((uint64_t)4, (uint64_t)4)))) {
            Binary_1[((iter5 - 0) + (max((uint64_t)4, (uint64_t)4) * (iter4 - 0)))] = ((Placeholder_1[(((4 == 1) ? 0 : ((4 == 1) ? 0 : iter5)) + (4 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter4))))] + (Placeholder_0[((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter4))] + 0)) - Placeholder_1[(((4 == 1) ? 0 : iter5) + (4 * ((256 == 1) ? 0 : iter4)))]);
        }
    }
}
