import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys

from parrots import save_json_temp

@pat(coderize=True, full_shape=True)
def local_int(x, y, w):
    z = x + y
    t = z[:,2] - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    y = torch.randn(M,N).cuda()
    sargs_list = []

    sargs = [x, y, w]
    fast_func = local_int
    assert parrots.allclose(fast_func(*sargs), fast_func._pyfunc(*sargs),
            equal_nan=True)
