
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var1, float* __restrict__ Var2, float* __restrict__ Var0, float* __restrict__ Var9) {
    const uint64_t iter4_iter5_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter5 = (iter4_iter5_fused % 1);
    const uint64_t iter4 = iter4_iter5_fused;
    if ((iter4 < 256)) {
        if ((iter5 < 1)) {
            Var9[(iter5 + iter4)] = (((1 == 1) ? Var0[iter4] : (Var1[((iter5 * 0) + iter4)] + Var2[((iter5 * 0) + iter4)])) * Var0[(iter5 + iter4)]);
        }
    }
}

