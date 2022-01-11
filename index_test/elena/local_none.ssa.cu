#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Placeholder_2, float* __restrict__ Binary_1) {
    const uint64_t iter3_iter4_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter4 = ((iter3_iter4_fused % 4) + 0);
    const uint64_t iter3 = ((iter3_iter4_fused / 4) + 0);
    if ((iter3 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)256)))) {
        if ((iter4 < (0 + 4))) {
            Binary_1[((iter4 - 0) + (4 * (iter3 - 0)))] = (((Placeholder_0[((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter3))] + Placeholder_1[((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter3))]) + 0) - Placeholder_2[(((4 == 1) ? 0 : iter4) + (4 * ((256 == 1) ? 0 : iter3)))]);
        }
    }
}

