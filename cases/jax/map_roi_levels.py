import time

import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jit


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
    scale = jnp.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = jnp.floor(jnp.log2(scale / finest_scale + 1e-6))
    # target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()     # jax没有clamp接口
    target_lvls = target_lvls.clip(a_min=0, a_max=num_levels - 1).astype(jnp.int32)

    return target_lvls


def main():
    rois = np.random.normal(size=(20, 5)).astype(np.float32)

    jax_func = jit(map_roi_levels)
    result = jax_func(rois)
    target = map_roi_levels(rois)
    assert jnp.allclose(target, result) 

    time_begin = time.time()
    for i in range(10000):
        map_roi_levels(rois).block_until_ready()
    time_mid = time.time()
    for i in range(10000):
        jax_func(rois).block_until_ready()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_begin))
    print("time costing jax: " + str(time_end - time_mid))


if __name__ == "__main__":
    main()