import time
import torch

def local_slice(x, y):
    z = x + y
    t = z[:,0:8:2] - z[:,1:9:2]
    return t


@torch.jit.script
def fast_local_slice(x, y):
    z = x + y
    t = z[:,0:8:2] - z[:,1:9:2]
    return t


if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    c = torch.randn(M, K).cuda()
    d = torch.randn(M, K).cuda()

    sargs = [c, d]

    time1 = time.time()
    for i in range(1000):
        ret = local_slice(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_local_slice(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
