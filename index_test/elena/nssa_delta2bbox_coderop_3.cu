
#include "elena_int.h"
extern "C" __global__ void coderop_3(float* __restrict__ Var1, float* __restrict__ Var0, float* __restrict__ Var2, float* __restrict__ Var8) {
    const uint64_t iter18_iter19_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter19 = (iter18_iter19_fused % 1);
    const uint64_t iter18 = iter18_iter19_fused;
    if ((iter18 < 3000)) {
        if ((iter19 < 1)) {
            Var2[(iter19 + iter18)] = Var1[(iter19 + iter18)];
        }
    }
    const uint64_t iter21 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter21 < 3000)) {
        Var8[iter21] = ((Var0[((iter21 * 4) + 3)] - Var0[((iter21 * 4) + 1)]) + 1);
    }
}

