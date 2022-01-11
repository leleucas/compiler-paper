import jax.numpy as jnp
from jax import jit, device_put
from jax import random
import numpy as np
import time

def local_int(x, y, w):
    z = x + y
    t = z[:,2] - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    x = np.random.randn(M,N).astype(np.float32)
    w = np.random.randn(M).astype(np.float32)
    y = np.random.randn(M,N).astype(np.float32)
    x = device_put(x)
    w = device_put(w)
    y = device_put(y)

    sargs = [x, y, w]
    fast_func = jit(local_int)

    time1 = time.time()
    for i in range(1000):
        local_int(*sargs).block_until_ready()
    time2 = time.time()

    print("jax gpu time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        fast_func(*sargs).block_until_ready()
    time2 = time.time()

    print("jax jit time: ", time2 - time1)
