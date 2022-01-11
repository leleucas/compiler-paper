
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var2, float* __restrict__ Var7) {
    const uint64_t iter5_iter6_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter6 = (iter5_iter6_fused % 4);
    const uint64_t iter5 = (iter5_iter6_fused / 4);
    if ((iter5 < 256)) {
        if ((iter6 < 4)) {
            Var7[(iter6 + (4 * iter5))] = (((Var0[(iter6 + (4 * iter5))] + Var1[(iter6 + (4 * iter5))]) - Var1[(iter6 + (4 * iter5))]) + (((1 == 1) && (iter6 == 3)) ? Var2[iter5] : (Var0[(iter6 + (4 * iter5))] + Var1[(iter6 + (4 * iter5))])));
        }
    }
}

