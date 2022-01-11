import time
import torch

def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def bbox2offset(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    
    assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * torch.log(gt_widths / ex_widths)
    offset_dh = wh * torch.log(gt_heights / ex_heights)
    offset = torch.stack((offset_dx, offset_dy, offset_dw, offset_dh), dim=1)
    return offset

@torch.jit.script
def fast_bbox2offset(boxes, gt_boxes, weights: tuple):
    assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * torch.log(gt_widths / ex_widths)
    offset_dh = wh * torch.log(gt_heights / ex_heights)
    offset = torch.stack((offset_dx, offset_dy, offset_dw, offset_dh), dim=1)
    return offset

if __name__ == "__main__":
    if(len(sys.argv) == 3):
        M = int(sys.argv[1])
        N = int(sys.argv[2])

    else:
        M = 3000
        N = 4
   
    boxes = torch.randn(M, N).cuda()
    gt = torch.randn(M, N).cuda()
    weights = (1.0, 1.0, 1.0, 1.0)
    
    sargs_list = [boxes, gt, weights]
    assert torch.allclose(
        bbox2offset(*sargs_list), fast_bbox2offset(*sargs_list), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = bbox2offset(*sargs_list)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = fast_bbox2offset(*sargs_list)

    torch.cuda.synchronize()
    time_end = time.time()


    print("time costing pytorch: " + str(time_mid - time_before))
    print("time costing torchscript: " + str(time_end - time_mid))
