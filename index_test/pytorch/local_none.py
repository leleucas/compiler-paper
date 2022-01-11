import time
import torch


def local_none(x, u, v):
    z = u + v
    t = z[:,None] - x
    return t


@torch.jit.script
def fast_local_none(x, u, v):
    z = u + v
    t = z[:,None] - x
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    u = torch.randn(M).cuda()

    sargs = [x, w, u]

    time1 = time.time()
    for i in range(1000):
        ret = local_none(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_none(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
