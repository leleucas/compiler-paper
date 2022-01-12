import tensorflow as tf
import time
import numpy as np


def global_int(x, w):
    z = x[:,1] + w
    t = z - w
    return t

if __name__ == "__main__":
    M = 256
    N = 4
    K = 9
    P = 1

    x = tf.random.uniform((M,N))
    w = tf.random.uniform((M,))

    sargs = [x, w]
    fast_func = tf.function(global_int, experimental_compile=True)

    time1 = time.time()
    for i in range(1000):
        ret = global_int(*sargs).numpy()
    time2 = time.time()

    print("tensorflow time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_func(*sargs).numpy()
    time2 = time.time()

    print("xla time: ", time2 - time1)
