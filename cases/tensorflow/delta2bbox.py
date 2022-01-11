import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    # if isinstance(deltas, torch.cuda.HalfTensor):
    #     deltas = deltas.float()
    means = tf.constant(means)
    means = tf.expand_dims(means,0)
    means = tf.tile(means, (tf.shape(deltas)[0] ,1))
    stds = tf.constant(stds)
    stds = tf.expand_dims(stds,0)
    stds = tf.tile(stds, (tf.shape(deltas)[0],1))
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = tf.clip_by_value(dw, -max_ratio, max_ratio)
    dh = tf.clip_by_value(dh, -max_ratio, max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    px = tf.expand_dims(px, 1)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    py = tf.expand_dims(py, 1)
    pw = (rois[:, 2] - rois[:, 0] + 1.0)
    pw = tf.expand_dims(pw, 1)
    ph = (rois[:, 3] - rois[:, 1] + 1.0)
    ph = tf.expand_dims(ph, 1)
    gw = pw * tf.math.exp(dw)
    gh = ph * tf.math.exp(dh)
    # gx = tf.math.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    # gy = tf.math.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gx = px + pw * dx
    gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = tf.clip_by_value(x1, 0, max_shape[1] - 1)
        y1 = tf.clip_by_value(y1, 0, max_shape[0] - 1)
        x2 = tf.clip_by_value(x2, 0, max_shape[1] - 1)
        y2 = tf.clip_by_value(y2, 0, max_shape[0] - 1)
    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    bboxes = tf.reshape(bboxes, deltas.shape)
    return bboxes

@tf.function(experimental_compile=True)
def fast_delta(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    # if isinstance(deltas, torch.cuda.HalfTensor):
    #     deltas = deltas.float()
    means = tf.constant(means)
    means = tf.expand_dims(means,0)
    means = tf.tile(means, (tf.shape(deltas)[0] ,1))
    stds = tf.constant(stds)
    stds = tf.expand_dims(stds,0)
    stds = tf.tile(stds, (tf.shape(deltas)[0],1))
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = tf.clip_by_value(dw, -max_ratio, max_ratio)
    dh = tf.clip_by_value(dh, -max_ratio, max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    px = tf.expand_dims(px, 1)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    py = tf.expand_dims(py, 1)
    pw = (rois[:, 2] - rois[:, 0] + 1.0)
    pw = tf.expand_dims(pw, 1)
    ph = (rois[:, 3] - rois[:, 1] + 1.0)
    ph = tf.expand_dims(ph, 1)
    gw = pw * tf.math.exp(dw)
    gh = ph * tf.math.exp(dh)
    # gx = tf.math.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    # gy = tf.math.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gx = px + pw * dx
    gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = tf.clip_by_value(x1, 0, max_shape[1] - 1)
        y1 = tf.clip_by_value(y1, 0, max_shape[0] - 1)
        x2 = tf.clip_by_value(x2, 0, max_shape[1] - 1)
        y2 = tf.clip_by_value(y2, 0, max_shape[0] - 1)
    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    bboxes = tf.reshape(bboxes, deltas.shape)
    return bboxes


if __name__ == "__main__":
    # data prepare
    M = 3000
    N = 4
    rois = tf.random.uniform((M,N))
    deltas = tf.random.uniform((M,N))
    means=[0.0, 0.0, 0.0, 0.0]
    stds=[1.0, 1.0, 1.0, 1.0]
    max_shape = (487, 480, 3)
    
    sargs = [rois, deltas, means, stds, max_shape]

    # correctness check
    assert np.allclose(delta2bbox(*sargs), fast_delta(*sargs), equal_nan=True)

    # performance check
    time_before = time.time()
    
    for _ in range(10000):
        ret = delta2bbox(*sargs)
    _ = ret.numpy()
    time_mid = time.time()
    
    for _ in range(10000):
        ret = fast_delta(*sargs)
    _ = ret.numpy()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_before))
    print("time costing coderized: " + str(time_end - time_mid))
    