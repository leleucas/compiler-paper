#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Placeholder_2, float* __restrict__ Binary_2) {
    const uint64_t iter6_iter7_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter7 = ((iter6_iter7_fused % 1) + 0);
    const uint64_t iter6 = ((iter6_iter7_fused / 1) + 0);
    if ((iter6 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)256)))) {
        if ((iter7 < (0 + 1))) {
            Binary_2[((iter7 - 0) + (1 * (iter6 - 0)))] = ((((Placeholder_2[(0 + (1 * ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter6)))] + 0) + 0) + 0) * Placeholder_2[(((1 == 1) ? 0 : iter7) + (1 * ((256 == 1) ? 0 : iter6)))]);
        }
    }
}

