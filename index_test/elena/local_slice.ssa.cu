#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Binary_1) {
    const uint64_t iter6_iter7_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter7 = ((iter6_iter7_fused % max((uint64_t)4, (uint64_t)4)) + 0);
    const uint64_t iter6 = ((iter6_iter7_fused / max((uint64_t)4, (uint64_t)4)) + 0);
    if ((iter6 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)max((uint64_t)256, (uint64_t)256))))) {
        if ((iter7 < (0 + max((uint64_t)4, (uint64_t)4)))) {
            Binary_1[((iter7 - 0) + (max((uint64_t)4, (uint64_t)4) * (iter6 - 0)))] = (((Placeholder_0[(((9 == 1) ? 0 : ((((4 == 1) ? 0 : iter7) * 2) + 0)) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter6))))] + Placeholder_1[(((9 == 1) ? 0 : ((((4 == 1) ? 0 : iter7) * 2) + 0)) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter6))))]) + 0) - ((Placeholder_0[(((9 == 1) ? 0 : ((((4 == 1) ? 0 : iter7) * 2) + 1)) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter6))))] + Placeholder_1[(((9 == 1) ? 0 : ((((4 == 1) ? 0 : iter7) * 2) + 1)) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter6))))]) + 0));
        }
    }
}

