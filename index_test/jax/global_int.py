import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys

from parrots import save_json_temp

@pat(coderize=True, full_shape=True)
def global_int(x, w):
    z = x[:,1] + w
    t = z - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    sargs_list = []

    sargs = [x, w]
    fast_func = global_int
    assert parrots.allclose(fast_func(*sargs), fast_func._pyfunc(*sargs),
            equal_nan=True)
