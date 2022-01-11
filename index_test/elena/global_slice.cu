
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var9) {
    const uint64_t iter2_iter3_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter3 = (iter2_iter3_fused % 5);
    const uint64_t iter2 = (iter2_iter3_fused / 5);
    if ((iter2 < 256)) {
        if ((iter3 < 5)) {
            Var9[(iter3 + (5 * iter2))] = ((Var0[((iter3 * 2) + (9 * iter2))] + Var1[((iter3 * 2) + (9 * iter2))]) - Var0[((iter3 * 2) + (9 * iter2))]);
        }
    }
}

