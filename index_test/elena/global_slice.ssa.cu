#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Binary_1) {
    const uint64_t iter8_iter9_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter9 = ((iter8_iter9_fused % max((uint64_t)max((uint64_t)5, (uint64_t)5), (uint64_t)5)) + 0);
    const uint64_t iter8 = ((iter8_iter9_fused / max((uint64_t)max((uint64_t)5, (uint64_t)5), (uint64_t)5)) + 0);
    if ((iter8 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)256)))) {
        if ((iter9 < (0 + max((uint64_t)max((uint64_t)5, (uint64_t)5), (uint64_t)5)))) {
            Binary_1[((iter9 - 0) + (max((uint64_t)max((uint64_t)5, (uint64_t)5), (uint64_t)5) * (iter8 - 0)))] = (((Placeholder_0[(((((5 == 1) ? 0 : ((max((uint64_t)5, (uint64_t)5) == 1) ? 0 : iter9)) * 2) + 0) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter8))))] + 0) + (Placeholder_1[(((((5 == 1) ? 0 : ((max((uint64_t)5, (uint64_t)5) == 1) ? 0 : iter9)) * 2) + 0) + (9 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter8))))] + 0)) - (Placeholder_0[(((((5 == 1) ? 0 : iter9) * 2) + 0) + (9 * ((256 == 1) ? 0 : iter8)))] + 0));
        }
    }
}

