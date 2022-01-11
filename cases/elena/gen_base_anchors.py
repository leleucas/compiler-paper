import time
import torch
import parrots
from parrots.jit import pat
import numpy as np
import sys


@pat(coderize=True)
# def gen_base_anchors(scales, ratios, base_size):
def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        dim=-1).round()

    return base_anchors

if __name__ == "__main__":

    # M denotes the number of anchors
    # N means the four coordinates, so N and K can be an arbitary number that larger than 3
    if(len(sys.argv) == 4):
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])


    else:
        M = 4
        N = 3
        K = 1
   
    base_size = M
    ratios = torch.randn(N).cuda()
    scales = torch.randn(K).cuda()


    sfunc = gen_base_anchors
    sargs_list = [base_size, ratios, scales]
    assert parrots.allclose(
        sfunc(*sargs_list), sfunc._pyfunc(*sargs_list), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = sfunc._pyfunc(*sargs_list)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = sfunc(*sargs_list)

    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_before))
    print("time costing coderized: " + str(time_end - time_mid))
