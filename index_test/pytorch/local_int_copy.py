import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys

from parrots import save_json_temp

@pat(coderize=True, full_shape=True)
def local_int_copy(x, y, w):
    z = x + y
    m = z - y
    z[:,3] = w
    t = m + z
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = torch.randn(M,N).cuda()
    y = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    sargs_list = []

    sargs = [x, y, w]
    fast_func = local_int_copy
    assert parrots.allclose(fast_func(*sargs), fast_func._pyfunc(*sargs),
            equal_nan=True)
