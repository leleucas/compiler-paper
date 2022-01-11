#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Binary_1) {
    const uint64_t iter2 = (((blockIdx.x * 64) + threadIdx.x) + 0);
    if ((iter2 < (0 + max((uint64_t)max((uint64_t)256, (uint64_t)256), (uint64_t)256)))) {
        Binary_1[(iter2 - 0)] = (((Placeholder_0[(1 + (4 * ((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter2))))] + 0) + Placeholder_1[((256 == 1) ? 0 : ((max((uint64_t)256, (uint64_t)256) == 1) ? 0 : iter2))]) - Placeholder_1[((256 == 1) ? 0 : iter2)]);
    }
}

