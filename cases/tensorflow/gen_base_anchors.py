import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


@tf.function(experimental_compile=True)
def fast_gen(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = tf.math.sqrt(ratios)
    w_ratios = 1 / h_ratios
    # ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
    # hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
    ws = w * w_ratios[:, None] * scales[None, :]
    ws = tf.reshape(ws, [-1])
    hs = h * h_ratios[:, None] * scales[None, :]
    hs = tf.reshape(hs, [-1])

    # base_anchors = tf.stack(
    #     [
    #         x_ctr - 0.5 * (ws - 1),
    #         y_ctr - 0.5 * (hs - 1),
    #         x_ctr + 0.5 * (ws - 1),
    #         y_ctr + 0.5 * (hs - 1),
    #     ],
    #     axis=-1,
    # ).round()
    base_anchors = tf.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        axis=-1,
    )
    base_anchors = tf.math.round(base_anchors)

    return base_anchors


def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = tf.math.sqrt(ratios)
    w_ratios = 1 / h_ratios
    # ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
    # hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
    ws = w * w_ratios[:, None] * scales[None, :]
    ws = tf.reshape(ws, [-1])
    hs = h * h_ratios[:, None] * scales[None, :]
    hs = tf.reshape(hs, [-1])

    # base_anchors = tf.stack(
    #     [
    #         x_ctr - 0.5 * (ws - 1),
    #         y_ctr - 0.5 * (hs - 1),
    #         x_ctr + 0.5 * (ws - 1),
    #         y_ctr + 0.5 * (hs - 1),
    #     ],
    #     axis=-1,
    # ).round()
    base_anchors = tf.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        axis=-1,
    )
    base_anchors = tf.math.round(base_anchors)

    return base_anchors


if __name__ == "__main__":
    M = 4
    N = 3
    K = 1

    base_size = M
    ratios = tf.ones([N], tf.float32)
    scales = tf.ones([K], tf.float32)

    sargs = [base_size, ratios, scales]
    assert np.allclose(fast_gen(*sargs).numpy(), gen_base_anchors(*sargs).numpy())

    time_begin = time.time()
    for _ in range(10000):
        a = gen_base_anchors(*sargs)
    _ = a.numpy()
    time_mid = time.time()
    for _ in range(10000):
        b = fast_gen(*sargs)
    _ = b.numpy()
    time_end = time.time()

    print("time tensorflow: " + str(time_mid - time_begin))
    print("time xla: " + str(time_end - time_mid))
