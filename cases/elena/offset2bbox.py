import numpy as np
import time
import torch
import parrots
from parrots import jit

def allow_empty_tensor(num=1, empty_shape=(0, 4)):
    """Return an empty tensor directly if any of first `num` argument is empty"""

    def decorate(func):
        def wrapper(*args, **kwargs):
            for arg in args[:num]:
                if torch.is_tensor(arg) and arg.numel() == 0:
                    return arg.new_zeros(empty_shape)
            return func(*args, **kwargs)

        return wrapper

    return decorate



@allow_empty_tensor(1)
def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return torch.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h

# TODO(limaolin): coderized func returns different value now.
@jit.pat(coderize=True)
@allow_empty_tensor(2)
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
    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.new_zeros(offset.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


if __name__ == "__main__":
    M = 4000
    N = 4
    K = 4
   
    boxes = torch.randn(M, N).cuda()
    offset = torch.randn(M, K).cuda()
    weights = (1.0, 1.0, 1.0, 1.0)


    sfunc = offset2bbox
    sargs_list = [boxes, offset, weights]
    assert parrots.allclose(
        sfunc(*sargs_list), sfunc._pyfunc(*sargs_list), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = sfunc._pyfunc(*sargs_list)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = sfunc(*sargs_list)

    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_before))
    print("time costing coderized: " + str(time_end - time_mid))
