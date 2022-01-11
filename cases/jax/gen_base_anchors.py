import jax.numpy as jnp
from jax import jit
import time

def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = jnp.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = jnp.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        axis=-1).round()

    return base_anchors


if __name__ == "__main__":
    M = 4
    N = 3
    K = 1

    base_size = M
    ratios_np = jnp.ones((N))
    scales_np = jnp.ones((K))

    fast_gen=jit(gen_base_anchors)
    sargs=(base_size, ratios_np, scales_np)
    assert jnp.allclose(gen_base_anchors(*sargs), fast_gen(*sargs))
    
    time_begin = time.time()
    for i in range(10000):
        gen_base_anchors(*sargs).block_until_ready()
    time_mid = time.time()
    for i in range(10000):
        fast_gen(*sargs).block_until_ready()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_begin))
    print("time costing jax: " + str(time_end - time_mid))
