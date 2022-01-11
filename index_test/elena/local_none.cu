
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var1, float* __restrict__ Var2, float* __restrict__ Var0, float* __restrict__ Var6) {
    const uint64_t iter1_iter2_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter2 = (iter1_iter2_fused % 4);
    const uint64_t iter1 = (iter1_iter2_fused / 4);
    if ((iter1 < 256)) {
        if ((iter2 < 4)) {
            Var6[(iter2 + (4 * iter1))] = ((Var1[((iter2 * 0) + iter1)] + Var2[((iter2 * 0) + iter1)]) - Var0[(iter2 + (4 * iter1))]);
        }
    }
}

