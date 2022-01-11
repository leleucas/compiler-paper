
#include "elena_int.h"
extern "C" __global__ void coderop_4(float* __restrict__ Var7, float* __restrict__ Var2, float* __restrict__ Var6, float* __restrict__ Var3, float* __restrict__ Var4, float* __restrict__ Var0, float* __restrict__ Var5, float* __restrict__ Var1, float* __restrict__ Var27, float* __restrict__ Var28, float* __restrict__ Var29, float* __restrict__ Var30) {
    const uint64_t iter60_iter61_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter61 = (iter60_iter61_fused % 1);
    const uint64_t iter60 = iter60_iter61_fused;
    if ((iter60 < 3000)) {
        if ((iter61 < 1)) {
            Var27[(iter61 + iter60)] = fmin((float)fmax((float)(((Var4[(iter61 + iter60)] + (Var6[(iter61 + iter60)] * Var0[(iter61 + iter60)])) - ((Var6[(iter61 + iter60)] * (expf(Var2[(iter61 + (1 * iter60))]))) * 0.5)) + 0.5), (float)0), (float)479);
        }
    }
    const uint64_t iter62_iter63_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter63 = (iter62_iter63_fused % 1);
    const uint64_t iter62 = iter62_iter63_fused;
    if ((iter62 < 3000)) {
        if ((iter63 < 1)) {
            Var28[(iter63 + iter62)] = fmin((float)fmax((float)(((Var5[(iter63 + iter62)] + (Var7[(iter63 + iter62)] * Var1[(iter63 + iter62)])) - ((Var7[(iter63 + iter62)] * (expf(Var3[(iter63 + (1 * iter62))]))) * 0.5)) + 0.5), (float)0), (float)486);
        }
    }
    const uint64_t iter64_iter65_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter65 = (iter64_iter65_fused % 1);
    const uint64_t iter64 = iter64_iter65_fused;
    if ((iter64 < 3000)) {
        if ((iter65 < 1)) {
            Var29[(iter65 + iter64)] = fmin((float)fmax((float)(((Var4[(iter65 + iter64)] + (Var6[(iter65 + iter64)] * Var0[(iter65 + iter64)])) + ((Var6[(iter65 + iter64)] * (expf(Var2[(iter65 + (1 * iter64))]))) * 0.5)) - 0.5), (float)0), (float)479);
        }
    }
    const uint64_t iter66_iter67_fused = ((blockIdx.x * 64) + threadIdx.x);
    const uint64_t iter67 = (iter66_iter67_fused % 1);
    const uint64_t iter66 = iter66_iter67_fused;
    if ((iter66 < 3000)) {
        if ((iter67 < 1)) {
            Var30[(iter67 + iter66)] = fmin((float)fmax((float)(((Var5[(iter67 + iter66)] + (Var7[(iter67 + iter66)] * Var1[(iter67 + iter66)])) + ((Var7[(iter67 + iter66)] * (expf(Var3[(iter67 + (1 * iter66))]))) * 0.5)) - 0.5), (float)0), (float)486);
        }
    }
}

