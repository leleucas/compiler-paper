import jax.numpy as jnp
from jax import jit, device_put
from jax import random
import numpy as np
import time

def local_slice(x, y):
    z = x + y
    t = z[:,0:8:2] - z[:,1:9:2]
    return t


if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    # global int
    c = np.random.randn(M, K).astype(np.float32)
    d = np.random.randn(M, K).astype(np.float32)
    c = device_put(c)
    d = device_put(d)

    sargs = [c, d]
    fast_func = jit(local_slice)

    time1 = time.time()
    for i in range(1000):
        local_slice(*sargs).block_until_ready()
    time2 = time.time()

    print("jax gpu time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        fast_func(*sargs).block_until_ready()
    time2 = time.time()

    print("jax jit time: ", time2 - time1)
