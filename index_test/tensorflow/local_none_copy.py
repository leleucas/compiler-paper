import tensorflow as tf
import time
import numpy as np


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

    w = tf.random.uniform((M,))
    u = tf.random.uniform((M,))
    a = tf.random.uniform((M,P))

    sargs = [a, w, u]
    fast_func = tf.function(local_none_copy, experimental_compile=True)

    time1 = time.time()
    for i in range(1000):
        ret = local_none_copy(*sargs).numpy()
    time2 = time.time()

    print("tensorflow time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_func(*sargs).numpy()
    time2 = time.time()

    print("xla time: ", time2 - time1)
