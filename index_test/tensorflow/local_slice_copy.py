import tensorflow as tf
import time
import numpy as np

def local_slice_copy(x, y):
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

    c = tf.random.uniform((M, K))
    d = tf.random.uniform((M, K))

    sargs = [c, d]
    fast_func = tf.function(local_slice_copy, experimental_compile=True)

    time1 = time.time()
    for i in range(1000):
        ret = local_slice_copy(*sargs).numpy()
    time2 = time.time()

    print("tensorflow time: ", time2 - time1)

    time1 = time.time()
    for i in range(1000):
        ret = fast_func(*sargs).numpy()
    time2 = time.time()

    print("xla time: ", time2 - time1)
