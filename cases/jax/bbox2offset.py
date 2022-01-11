import jax.numpy as jnp
from jax import jit
import time


def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def bbox2offset(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * jnp.log(gt_widths / ex_widths)
    offset_dh = wh * jnp.log(gt_heights / ex_heights)
    offset = jnp.stack((offset_dx, offset_dy, offset_dw, offset_dh), axis=1)
    return offset

if __name__ == "__main__":
    N = 3000
    K = 4

    bboxes_np = jnp.ones((N,K))
    offsets_np = jnp.ones((N,K))
    weight_np = jnp.ones(K)

    fast_bbox=jit(bbox2offset)
    sargs=(bboxes_np, offsets_np, weight_np)
    assert jnp.allclose(bbox2offset(*sargs), fast_bbox(*sargs))
    
    time_begin = time.time()
    for i in range(10000):
        bbox2offset(bboxes_np, offsets_np, weight_np).block_until_ready()
    time_mid = time.time()
    for i in range(10000):
        fast_bbox(bboxes_np, offsets_np, weight_np).block_until_ready()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_begin))
    print("time costing jax: " + str(time_end - time_mid))
