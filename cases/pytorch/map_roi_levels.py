import time
import torch


def map_roi_levels(rois, num_levels=4, finest_scale=56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = torch.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long() 

    return target_lvls


@torch.jit.script
def fast_map_roi_levels(rois, num_levels=4, finest_scale=56):
    # type: (Tensor, int, int) -> Tensor
    scale = torch.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long() 

    return target_lvls


def main():
    
    # rois_list = torch.load('./data/rois_list.pth')
    # rois = rois_list[0]       # (20, 5)
    shape = (20, 5)
    rois = torch.randn(shape, dtype=torch.float32, device='cuda')
    num_levels = 5
    finest_scale = 56
    sargs_list = [rois, num_levels, finest_scale]

    assert torch.allclose(
        map_roi_levels(*sargs_list), fast_map_roi_levels(*sargs_list), equal_nan=True)


    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = map_roi_levels(*sargs_list)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = fast_map_roi_levels(*sargs_list)

    torch.cuda.synchronize()
    time_end = time.time()


    print("time costing pytorch: " + str(time_mid - time_before))
    print("time costing torchscript: " + str(time_end - time_mid))


if __name__ == "__main__":
    main()