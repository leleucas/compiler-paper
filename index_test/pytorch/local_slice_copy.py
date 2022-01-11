import time
import torch


def local_slice_copy(x, y):
    z = x + y
    m = z - y
    z[:,0:9:2] = x[:,0:9:2] * y[:,0:9:2]
    t = m + z
    return t

@torch.jit.script
def fast_local_slice_copy(x, y):
    z = x + y
    m = z - y
    z[:,0:9:2] = x[:,0:9:2] * y[:,0:9:2]
    t = m + z
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = torch.randn(M,N).cuda()
    w = torch.randn(M).cuda()
    c = torch.randn(M, K).cuda()
    d = torch.randn(M, K).cuda()

    sargs = [c, d]

    time1 = time.time()
    for i in range(1000):
        ret = local_slice_copy(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_slice_copy(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
