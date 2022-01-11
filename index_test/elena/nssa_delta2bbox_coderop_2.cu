
#include "elena_int.h"
extern "C" __global__ void coderop_2(float* __restrict__ Var1, float* __restrict__ Var0, float* __restrict__ Var2, float* __restrict__ Var8) {
    const uint64_t iter14_iter15_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter15 = (iter14_iter15_fused % 1);
    const uint64_t iter14 = iter14_iter15_fused;
    if ((iter14 < 3000)) {
        if ((iter15 < 1)) {
            Var2[(iter15 + iter14)] = Var1[(iter15 + iter14)];
        }
    }
    const uint64_t iter17 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter17 < 3000)) {
        Var8[iter17] = ((Var0[((iter17 * 4) + 2)] - Var0[(iter17 * 4)]) + 1);
    }
}

