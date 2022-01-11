import torch
import parrots
from parrots import jit

import time


@jit.pat(coderize=True)
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


if __name__ == "__main__":

    # 使用下面这种数据构造会出现 nan error
    M = 20
    N = 5
    shape = (M, N)
    rois = torch.zeros(shape, dtype=torch.float32, device='cuda')
    rois[:,3]=torch.ones((M,), dtype=torch.float32, device='cuda')
    
    num_levels=4
    finest_scale=56

    sfunc = map_roi_levels
    sargs = [rois, num_levels, finest_scale]
    assert parrots.allclose(sfunc(*sargs), sfunc._pyfunc(*sargs), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()
    for _ in range(10000):
        result = sfunc._pyfunc(*sargs)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        result = sfunc(*sargs)

    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_before))
    print("time costing coderized: " + str(time_end - time_mid))

