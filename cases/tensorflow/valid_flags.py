# import tensorflow.compat.v1 as tf
import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def meshgrid(x, y, row_major=True):
    # xx = x.repeat(len(y))
    xx = tf.repeat(x, len(y), 0)
    # yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    yy = tf.reshape(y, [-1, 1])
    yy = tf.repeat(yy, len(x), 1)
    yy = tf.reshape(yy, [-1])
    if row_major:
        return xx, yy
    else:
        return yy, xx


def valid_flags(featmap_size, valid_size, num_base_anchors):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = tf.zeros(feat_w, dtype=tf.uint8)
    valid_y = tf.zeros(feat_h, dtype=tf.uint8)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    # valid = valid[:, None].expand(
    #     valid.size(0), num_base_anchors).contiguous().reshape(-1)
    valid = valid[:, None]
    valid = tf.repeat(valid, num_base_anchors, axis=-1)
    # valid = tf.contiguous(valid)
    valid = tf.reshape(valid, [-1])
    return valid


@tf.function(experimental_compile=True)
def fast_valid(featmap_size, valid_size, num_base_anchors):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = tf.zeros(feat_w, dtype=tf.uint8)
    valid_y = tf.zeros(feat_h, dtype=tf.uint8)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    # valid = valid[:, None].expand(
    #     valid.size(0), num_base_anchors).contiguous().reshape(-1)
    valid = valid[:, None]
    valid = tf.repeat(valid, num_base_anchors, axis=-1)
    # valid = tf.contiguous(valid)
    valid = tf.reshape(valid, [-1])
    return valid


if __name__ == "__main__":

    M = 160
    N = 120
    K = 3

    featmap_size = (M, N)
    valid_size = (M, N)
    num_base_anchors = K

    sargs = (featmap_size, valid_size, num_base_anchors)
    assert np.allclose(fast_valid(*sargs).numpy(), valid_flags(*sargs).numpy())

    time_begin = time.time()
    for _ in range(10000):
        a = valid_flags(*sargs)
    _ = a.numpy()
    time_mid = time.time()
    for _ in range(10000):
        b = fast_valid(*sargs)
    _ = b.numpy()
    time_end = time.time()

    print("time tensorflow: " + str(time_mid - time_begin))
    print("time xla: " + str(time_end - time_mid))
