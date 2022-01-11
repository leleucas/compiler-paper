import time
import torch


def global_slice(x, y):
    z = x[:,0:9:2] + y[:,0:9:2]
    t = z - x[:,0:9:2]
    return t


@torch.jit.script
def fast_global_slice(x, y):
    z = x[:,0:9:2] + y[:,0:9:2]
    t = z - x[:,0:9:2]
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
        ret = global_slice(*sargs)
    time2 = time.time()
    print("torch time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_global_slice(*sargs)
    time2 = time.time()
    print("torch jit time: ", time2 - time1)
