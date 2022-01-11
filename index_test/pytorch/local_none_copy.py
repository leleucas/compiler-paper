import time
import torch


def local_none_copy(x, u, v):
    z = u + v
    w = z - v
    z[:,None] = x
    t = z[:,None] * x
    return t

@torch.jit.script
def fast_local_none_copy(x, u, v):
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

    sargs = [a, w, u]

    time1 = time.time()
    for i in range(1000):
        ret = local_none_copy(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_none_copy(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
