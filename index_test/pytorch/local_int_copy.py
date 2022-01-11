import time
import torch


def local_int_copy(x, y, w):
    z = x + y
    m = z - y
    z[:,3] = w
    t = m + z
    return t


@torch.jit.script
def fast_local_int_copy(x, y, w):
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

    sargs = [x, y, w]

    time1 = time.time()
    for i in range(1000):
        ret = local_int_copy(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_int_copy(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
