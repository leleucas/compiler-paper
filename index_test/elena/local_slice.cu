
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var7) {
    const uint64_t iter2_iter3_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter3 = (iter2_iter3_fused % 4);
    const uint64_t iter2 = (iter2_iter3_fused / 4);
    if ((iter2 < 256)) {
        if ((iter3 < 4)) {
            Var7[(iter3 + (4 * iter2))] = ((Var0[((iter3 * 2) + (9 * iter2))] + Var1[((iter3 * 2) + (9 * iter2))]) - (Var0[(((iter3 * 2) + (9 * iter2)) + 1)] + Var1[(((iter3 * 2) + (9 * iter2)) + 1)]));
        }
    }
}

