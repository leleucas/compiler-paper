import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys

from parrots import save_json_temp

@pat(coderize=True, full_shape=True)
def local_slice_copy(x, y):
    z = x + y
    m = z - y
    z[:,0:9:2] = x[:,0:9:2] * y[:,0:9:2]
    t = m + z
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    c = torch.randn(M, K).cuda()
    d = torch.randn(M, K).cuda()
    sargs_list = []

    sargs = [c, d]
    fast_func = local_slice_copy
    assert parrots.allclose(fast_func(*sargs), fast_func._pyfunc(*sargs),
            equal_nan=True)
