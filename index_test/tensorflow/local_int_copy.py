import tensorflow as tf
import time
import numpy as np

def local_int_copy(x, y, w):
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

    x = tf.random.uniform((M,N))
    y = tf.random.uniform((M,N))
    w = tf.random.uniform((M,))

    sargs = [x, y, w]
    fast_func = tf.function(local_int_copy, experimental_compile=True)

    time1 = time.time()
    for i in range(1000):
        ret = local_int_copy(*sargs).numpy()
    time2 = time.time()

    print("tensorflow time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_func(*sargs).numpy()
    time2 = time.time()

    print("xla time: ", time2 - time1)
