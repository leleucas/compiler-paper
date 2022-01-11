
#include "elena_int.h"
extern "C" __global__ void coderop_0(float* __restrict__ Var0, float* __restrict__ Var1, float* __restrict__ Var5) {
    const uint64_t iter1 = ((blockIdx.x * 64) + threadIdx.x);
    if ((iter1 < 256)) {
        Var5[iter1] = ((Var0[((iter1 * 4) + 1)] + Var1[iter1]) - Var1[iter1]);
    }
}

