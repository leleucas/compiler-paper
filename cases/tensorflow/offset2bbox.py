import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return tf.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h

def offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = tf.clip_by_value(dw, dw, np.log(1000. / 16.))
    dh = tf.clip_by_value(dh, dh, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = tf.exp(dw) * widths[:, None]
    pred_h = tf.exp(dh) * heights[:, None]

    pred_boxes = tf.Variable(tf.zeros(offset.shape))
    # x1
    pred_boxes[:, 0::4].assign(pred_ctr_x - 0.5 * pred_w)
    # y1
    pred_boxes[:, 1::4].assign(pred_ctr_y - 0.5 * pred_h)
    # x2
    pred_boxes[:, 2::4].assign(pred_ctr_x + 0.5 * pred_w - 1)
    # y2
    pred_boxes[:, 3::4].assign(pred_ctr_y + 0.5 * pred_h - 1)

    pred_boxes = tf.convert_to_tensor(pred_boxes)

    return pred_boxes

# @tf.function(experimental_compile=True)
def fast_offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = tf.clip_by_value(dw, dw, np.log(1000. / 16.))
    dh = tf.clip_by_value(dh, dh, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = tf.exp(dw) * widths[:, None]
    pred_h = tf.exp(dh) * heights[:, None]

    pred_boxes = tf.Variable(tf.zeros(offset.shape))
    # x1
    pred_boxes[:, 0::4].assign(pred_ctr_x - 0.5 * pred_w)
    # y1
    pred_boxes[:, 1::4].assign(pred_ctr_y - 0.5 * pred_h)
    # x2
    pred_boxes[:, 2::4].assign(pred_ctr_x + 0.5 * pred_w - 1)
    # y2
    pred_boxes[:, 3::4].assign(pred_ctr_y + 0.5 * pred_h - 1)

    pred_boxes = tf.convert_to_tensor(pred_boxes)

    return pred_boxes


if __name__ == "__main__":
    M = 3000
    N = 4
    boxes = tf.ones([M, N], tf.float32)
    offset = tf.ones([M, N], tf.float32)
    weights = [1.0, 1.0, 1.0, 1.0]
    
    sargs_list = [boxes, offset, weights]
    assert np.allclose(
        offset2bbox(*sargs_list).numpy(), fast_offset2bbox(*sargs_list).numpy())

    # performance check
    time_begin = time.time()

    for _ in range(10000):
        ret = offset2bbox(*sargs_list)
    _ = ret.numpy()
    time_mid = time.time()

    for _ in range(10000):
        ret = fast_offset2bbox(*sargs_list)
    _ = ret.numpy()
    time_end = time.time()

    print("time tensorflow: " + str(time_mid - time_begin))
    print("time xla: " + str(time_end - time_mid))
