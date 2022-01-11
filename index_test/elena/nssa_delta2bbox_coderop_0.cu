
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var1, float* __restrict__ Var3, float* __restrict__ Var2, float* __restrict__ Var0, float* __restrict__ Var5, float* __restrict__ Var14, float* __restrict__ Var15, float* __restrict__ Var21) {
    const uint64_t iter2_iter3_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter3 = (iter2_iter3_fused % 4);
    const uint64_t iter2 = (iter2_iter3_fused / 4);
    if ((iter2 < 3000)) {
        if ((iter3 < 4)) {
            Var5[(iter3 + (4 * iter2))] = ((Var1[(iter3 + (4 * iter2))] * Var3[(iter3 + (0 * iter2))]) + Var2[(iter3 + (0 * iter2))]);
        }
    }
    const uint64_t iter4_iter5_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter5 = (iter4_iter5_fused % 1);
    const uint64_t iter4 = iter4_iter5_fused;
    if ((iter4 < 3000)) {
        if ((iter5 < 1)) {
            Var14[(iter5 + iter4)] = fmin((float)fmax((float)((Var1[(((iter5 * 2) + (4 * iter4)) + 2)] * Var3[(((iter5 * 2) + (0 * iter4)) + 2)]) + Var2[(((iter5 * 2) + (0 * iter4)) + 2)]), (float)-4.13517), (float)4.13517);
        }
    }
    const uint64_t iter6_iter7_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter7 = (iter6_iter7_fused % 1);
    const uint64_t iter6 = iter6_iter7_fused;
    if ((iter6 < 3000)) {
        if ((iter7 < 1)) {
            Var15[(iter7 + iter6)] = fmin((float)fmax((float)((Var1[((iter7 + (4 * iter6)) + 3)] * Var3[((iter7 + (0 * iter6)) + 3)]) + Var2[((iter7 + (0 * iter6)) + 3)]), (float)-4.13517), (float)4.13517);
        }
    }
    const uint64_t iter9 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter9 < 3000)) {
        Var21[iter9] = ((Var0[(iter9 * 4)] + Var0[((iter9 * 4) + 2)]) * 0.5);
    }
}

