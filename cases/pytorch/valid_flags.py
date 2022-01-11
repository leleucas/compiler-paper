import time
import torch
import numpy as np
import sys

    

def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx

def valid_flags(featmap_size, valid_size, num_base_anchors, device='cuda'):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(
        valid.size(0), num_base_anchors).contiguous().view(-1)
    return valid

@torch.jit.script
def fast_valid(featmap_size: tuple, valid_size: tuple, num_base_anchors: int, device='cuda'):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(
        valid.size(0), num_base_anchors).contiguous().view(-1)
    return valid

if __name__ == "__main__":

    # M, N denotes the featmap size
    # K means the number of base anchors
    if(len(sys.argv) == 4):
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])

    else:
        M = 160
        N = 120
        K = 3
   
    featmap_size = (M, N)
    valid_size = (M, N)
    num_base_anchors = K

    sfunc = valid_flags
    sargs_list = [featmap_size, valid_size, num_base_anchors]
    assert torch.allclose(
        valid_flags(*sargs_list), fast_valid(*sargs_list), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = valid_flags(*sargs_list)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = fast_valid(*sargs_list)

    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing torch: " + str(time_mid - time_before))
    print("time costing torchscript: " + str(time_end - time_mid))
