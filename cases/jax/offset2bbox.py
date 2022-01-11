import jax
import jax.numpy as jnp
from jax import jit
import time

def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return jnp.stack((cx, cy, w, h), axis=1)
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
    dw = jax.lax.max(dw, jnp.log(1000. / 16.))
    dh = jax.lax.max(dh, jnp.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = jnp.exp(dw) * widths[:, None]
    pred_h = jnp.exp(dh) * heights[:, None]

    pred_boxes = jnp.zeros(offset.shape)
    jax.ops.index_update(pred_boxes, jax.ops.index[:, 0::4], pred_ctr_x - 0.5 * pred_w)
    jax.ops.index_update(pred_boxes, jax.ops.index[:, 1::4], pred_ctr_y - 0.5 * pred_h)
    jax.ops.index_update(pred_boxes, jax.ops.index[:, 2::4], pred_ctr_x + 0.5 * pred_w - 1)
    jax.ops.index_update(pred_boxes, jax.ops.index[:, 3::4], pred_ctr_y + 0.5 * pred_h - 1)

    return pred_boxes


if __name__ == "__main__":
    N = 3000
    K = 4

    bboxes = jnp.ones((N,K))
    offset = jnp.ones((N,K))
    weights = jnp.ones(K)

    fast_bbox=jit(offset2bbox)
    sargs=(bboxes, offset, weights)
    assert jnp.allclose(offset2bbox(*sargs), fast_bbox(*sargs))
    
    time_begin = time.time()
    for i in range(1000):
        offset2bbox(bboxes, offset, weights).block_until_ready()
    time_mid = time.time()
    for i in range(1000):
        fast_bbox(bboxes, offset, weights).block_until_ready()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_begin))
    print("time costing jax: " + str(time_end - time_mid))
