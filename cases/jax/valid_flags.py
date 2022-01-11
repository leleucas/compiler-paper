import jax.numpy as jnp
from jax import jit
import time

def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.reshape(-1, 1).tile(len(x)).reshape(-1) #np.tile <- jnp.repeat
    if row_major:
        return xx, yy
    else:
        return yy, xx

def valid_flags(featmap_size, valid_size, num_base_anchors):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = jnp.zeros(feat_w, dtype=jnp.uint8)
    valid_y = jnp.zeros(feat_h, dtype=jnp.uint8)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(
        valid.size(0), num_base_anchors).contiguous().reshape(-1)
    return valid


if __name__ == "__main__":
    M = 160
    N = 120
    K = 3

    featmap_size = (M, N)
    valid_size = (M, N)
    num_base_anchors = K

    fast_valid=jit(valid_flags)
    sargs=(featmap_size, valid_size, num_base_anchors)
    assert jnp.allclose(valid_flags(*sargs), fast_valid(*sargs))
    
    time_begin = time.time()
    for i in range(10000):
        valid_flags(*sargs).block_until_ready()
    time_mid = time.time()
    for i in range(10000):
        fast_valid(*sargs).block_until_ready()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_begin))
    print("time costing jax: " + str(time_end - time_mid))
