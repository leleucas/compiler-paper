import tensorflow as tf
import time
import numpy as np


def local_slice(x, y):
    z = x + y
    t = z[:,0:8:2] - z[:,1:9:2]
    return t


if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    c = tf.random.uniform((M, K))
    d = tf.random.uniform((M, K))

    sargs = [c, d]
    fast_func = tf.function(local_slice, experimental_compile=True)

    time1 = time.time()
    for i in range(1000):
        ret = local_slice(*sargs).numpy()
    time2 = time.time()

    print("tensorflow time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_func(*sargs).numpy()
    time2 = time.time()

    print("xla time: ", time2 - time1)
