import jax.numpy as jnp
from jax import jit, device_put
from jax import random
import numpy as np
import time


def local_none_copy(x, u, v):
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

    w = np.random.randn(M).astype(np.float32)
    u = np.random.randn(M).astype(np.float32)
    a = np.random.randn(M,P).astype(np.float32)
    w = device_put(w)
    a = device_put(a)
    u = device_put(u)

    sargs = [a, w, u]
    fast_func = jit(local_none_copy)

    time1 = time.time()
    for i in range(1000):
        local_none_copy(*sargs).block_until_ready()
    time2 = time.time()

    print("jax gpu time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        fast_func(*sargs).block_until_ready()
    time2 = time.time()

    print("jax jit time: ", time2 - time1)
