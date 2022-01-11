
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var11) {
    const uint64_t iter8_iter9_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter9 = (iter8_iter9_fused % 9);
    const uint64_t iter8 = (iter8_iter9_fused / 9);
    if ((iter8 < 256)) {
        if ((iter9 < 9)) {
            Var11[(iter9 + (9 * iter8))] = (((Var0[(iter9 + (9 * iter8))] + Var1[(iter9 + (9 * iter8))]) - Var1[(iter9 + (9 * iter8))]) + (((((1 == 1) && (iter9 >= 0)) && (iter9 < 9)) && ((iter9 % 2) == 0)) ? (Var0[(((iter9 / 2) * 2) + (9 * iter8))] * Var1[(((iter9 / 2) * 2) + (9 * iter8))]) : (Var0[(iter9 + (9 * iter8))] + Var1[(iter9 + (9 * iter8))])));
        }
    }
}

