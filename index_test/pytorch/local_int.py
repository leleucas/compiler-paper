import time
import torch


def local_int(x, y, w):
    z = x + y
    t = z[:,2] - w
    return t


@torch.jit.script
def fast_local_int(x, y, w):
    z = x + y
    t = z[:,2] - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    y = torch.randn(M,N).cuda()

    sargs = [x, y, w]

    time1 = time.time()
    for i in range(1000):
        ret = local_int(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_int(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
