# import tensorflow.compat.v1 as tf
import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def get_center_coordinates_and_sizes(box_corners):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    ycenter = ymin + height * 0.5
    xcenter = xmin + width * 0.5
    return [ycenter, xcenter, height, width]


@tf.function(experimental_compile=True)
def fast_encode(boxes, anchors, scale_factors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(boxes)
    # Avoid NaN in division and log below.
    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.math.log(w / wa)
    th = tf.math.log(h / ha)
    # Scales location targets as used in paper for joint training.
    ty *= scale_factors[0]
    tx *= scale_factors[1]
    th *= scale_factors[2]
    tw *= scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))


def encode(boxes, anchors, scale_factors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(boxes)
    # Avoid NaN in division and log below.
    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.math.log(w / wa)
    th = tf.math.log(h / ha)
    # Scales location targets as used in paper for joint training.
    ty *= scale_factors[0]
    tx *= scale_factors[1]
    th *= scale_factors[2]
    tw *= scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))


if __name__ == "__main__":

    M = 3000
    N = 4
    boxes = tf.ones([M, N], tf.float32)
    anchors = tf.ones([M, N], tf.float32)

    scale_factors = [1.0, 1.0, 1.0, 1.0]

    sargs = [boxes, anchors, scale_factors]
    assert np.allclose(fast_encode(*sargs).numpy(), encode(*sargs).numpy())

    time_begin = time.time()
    for _ in range(10000):
        a = encode(*sargs)
    _ = a.numpy()
    time_mid = time.time()
    for _ in range(10000):
        b = fast_encode(*sargs)
    _ = b.numpy()
    time_end = time.time()

    print("time tensorflow: " + str(time_mid - time_begin))
    print("time xla: " + str(time_end - time_mid))
