import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys

from parrots import save_json_temp

@pat(coderize=True, full_shape=True)
def local_none_copy(x, u, v):
    z = u + v
    w = z - v
    z[:,None] = x
    t = z[:,None] * x
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    u = torch.randn(M).cuda()
    a = torch.randn(M,P).cuda()
    b = torch.randn(M,P).cuda()
    sargs_list = []

    sargs = [a, w, u]
    fast_func = local_none_copy
    assert parrots.allclose(fast_func(*sargs), fast_func._pyfunc(*sargs),
            equal_nan=True)
