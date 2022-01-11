
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Placeholder_0, float* __restrict__ Placeholder_1, float* __restrict__ Placeholder_2, float* __restrict__ Placeholder_3, float* __restrict__ Binary_4, float* __restrict__ Clamp_0, float* __restrict__ Clamp_1, float* __restrict__ Binary_3) {
    const uint64_t iter2_iter3_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter3 = ((iter2_iter3_fused % max((uint64_t)max((uint64_t)4, (uint64_t)4), (uint64_t)4)) + 0);
    const uint64_t iter2 = ((iter2_iter3_fused / max((uint64_t)max((uint64_t)4, (uint64_t)4), (uint64_t)4)) + 0);
    if ((iter2 < (0 + 3000))) {
        if ((iter3 < (0 + max((uint64_t)max((uint64_t)4, (uint64_t)4), (uint64_t)4)))) {
            Binary_4[((iter3 - 0) + (max((uint64_t)max((uint64_t)4, (uint64_t)4), (uint64_t)4) * (iter2 - 0)))] = ((Placeholder_0[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : iter3)) + (4 * ((3000 == 1) ? 0 : ((3000 == 1) ? 0 : iter2))))] * Placeholder_1[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : iter3)) + (4 * ((1 == 1) ? 0 : ((3000 == 1) ? 0 : iter2))))]) + Placeholder_2[(((4 == 1) ? 0 : iter3) + (4 * ((1 == 1) ? 0 : iter2)))]);
        }
    }
    const uint64_t iter10_iter11_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter11 = ((iter10_iter11_fused % 1) + 0);
    const uint64_t iter10 = ((iter10_iter11_fused / 1) + 0);
    if ((iter10 < (0 + 3000))) {
        if ((iter11 < (0 + 1))) {
            Clamp_0[((iter11 - 0) + (1 * (iter10 - 0)))] = fmax((float)fmin((float)((((Placeholder_0[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : ((iter11 * 2) + 2))) + (4 * ((3000 == 1) ? 0 : ((3000 == 1) ? 0 : iter10))))] * Placeholder_1[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : ((iter11 * 2) + 2))) + (4 * ((1 == 1) ? 0 : ((3000 == 1) ? 0 : iter10))))]) + Placeholder_2[(((4 == 1) ? 0 : ((iter11 * 2) + 2)) + (4 * ((1 == 1) ? 0 : iter10)))]) + 0) + 0), (float)4.13517), (float)-4.13517);
        }
    }
    const uint64_t iter16_iter17_fused = (((blockIdx.x * 64) + threadIdx.x) + 0);
    const uint64_t iter17 = ((iter16_iter17_fused % 1) + 0);
    const uint64_t iter16 = ((iter16_iter17_fused / 1) + 0);
    if ((iter16 < (0 + 3000))) {
        if ((iter17 < (0 + 1))) {
            Clamp_1[((iter17 - 0) + (1 * (iter16 - 0)))] = fmax((float)fmin((float)((((Placeholder_0[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : ((iter17 * 1) + 3))) + (4 * ((3000 == 1) ? 0 : ((3000 == 1) ? 0 : iter16))))] * Placeholder_1[(((4 == 1) ? 0 : ((max((uint64_t)4, (uint64_t)4) == 1) ? 0 : ((iter17 * 1) + 3))) + (4 * ((1 == 1) ? 0 : ((3000 == 1) ? 0 : iter16))))]) + Placeholder_2[(((4 == 1) ? 0 : ((iter17 * 1) + 3)) + (4 * ((1 == 1) ? 0 : iter16)))]) + 0) + 0), (float)4.13517), (float)-4.13517);
        }
    }
    const uint64_t iter26 = (((blockIdx.x * 64) + threadIdx.x) + 0);
    if ((iter26 < (0 + max((uint64_t)3000, (uint64_t)3000)))) {
        Binary_3[(iter26 - 0)] = (((Placeholder_3[(0 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter26))))] + 0) + (Placeholder_3[(2 + (4 * ((3000 == 1) ? 0 : ((max((uint64_t)3000, (uint64_t)3000) == 1) ? 0 : iter26))))] + 0)) * 0.5);
    }
}

