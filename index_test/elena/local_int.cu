
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var2, float* __restrict__ Var6) {
    const uint64_t iter2 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter2 < 256)) {
        Var6[iter2] = ((Var0[((iter2 * 4) + 2)] + Var1[((iter2 * 4) + 2)]) - Var2[iter2]);
    }
}

