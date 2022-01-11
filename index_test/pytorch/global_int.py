import time
import torch


def global_int(x, w):
    z = x[:,1] + w
    t = z - w
    return t


@torch.jit.script
def fast_global_int(x, w):
    z = x[:,1] + w
    t = z - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()

    sargs = [x, w]

    time1 = time.time()
    for i in range(1000):
        ret = global_int(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_global_int(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
