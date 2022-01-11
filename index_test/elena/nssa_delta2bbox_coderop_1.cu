
#include "elena_int.h"
extern "C" __global__ void coderop_1(float* __restrict__ Var1, float* __restrict__ Var0, float* __restrict__ Var2, float* __restrict__ Var8) {
    const uint64_t iter10_iter11_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter11 = (iter10_iter11_fused % 1);
    const uint64_t iter10 = iter10_iter11_fused;
    if ((iter10 < 3000)) {
        if ((iter11 < 1)) {
            Var2[(iter11 + iter10)] = Var1[(iter11 + iter10)];
        }
    }
    const uint64_t iter13 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter13 < 3000)) {
        Var8[iter13] = ((Var0[((iter13 * 4) + 1)] + Var0[((iter13 * 4) + 3)]) * 0.5);
    }
}

