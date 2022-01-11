import numpy as torch
import time
import torch
import sys
import numpy as np

def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    # if isinstance(deltas, torch.cuda.HalfTensor):
    #     deltas = deltas.float()
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes

# @torch.jit.script
# def fast_delta2bbox(rois,
#                deltas,
#                means,
#                stds,
#                max_shape=None,
#                wh_ratio_clip=16 / 1000):
#     # if isinstance(deltas, torch.cuda.HalfTensor):
#     #     deltas = deltas.float()
#     means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
#     stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
#     denorm_deltas = deltas * stds + means
#     dx = denorm_deltas[:, 0::4]
#     dy = denorm_deltas[:, 1::4]
#     dw = denorm_deltas[:, 2::4]
#     dh = denorm_deltas[:, 3::4]
#     max_ratio = np.abs(np.log(wh_ratio_clip))
#     dw = dw.clamp(min=-max_ratio, max=max_ratio)
#     dh = dh.clamp(min=-max_ratio, max=max_ratio)
#     px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
#     py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
#     pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
#     ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
#     gw = pw * dw.exp()
#     gh = ph * dh.exp()
#     gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
#     gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
#     x1 = gx - gw * 0.5 + 0.5
#     y1 = gy - gh * 0.5 + 0.5
#     x2 = gx + gw * 0.5 - 0.5
#     y2 = gy + gh * 0.5 - 0.5
#     if max_shape is not None:
#         x1 = x1.clamp(min=0, max=max_shape[1] - 1)
#         y1 = y1.clamp(min=0, max=max_shape[0] - 1)
#         x2 = x2.clamp(min=0, max=max_shape[1] - 1)
#         y2 = y2.clamp(min=0, max=max_shape[0] - 1)
#     bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
#     return bboxes


if __name__ == "__main__":
    
    # M denotes the number of anchors
    # N and K means the four coordinates, so N and K can be an arbitary number that larger than 3
    if(len(sys.argv) == 4):
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])

    else:
        M = 3000
        N = 4
        K = 4
    rois = torch.randn(M, N).cuda()
    deltas = torch.randn(M, K).cuda()
    means = torch.zeros(N,).cuda()
    stds = torch.ones(N,).cuda()
    max_shape = torch.randn(3).cuda()

    # fast_delta2bbox = jit(delta2bbox)
    sargs = [rois, deltas, means, stds, max_shape]

    # correctness check
    # assert torch.allclose(fast_delta2bbox(*sargs), delta2bbox(*sargs), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()
    
    for _ in range(10000):
        ret = delta2bbox(*sargs)
    
    torch.cuda.synchronize()
    time_mid = time.time()
    
    # for _ in range(10000):
    #     ret = fast_delta2bbox(*sargs)
    
    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing torch: " + str(time_mid - time_before))
    print("time costing torchScript: " + str(time_end - time_mid))