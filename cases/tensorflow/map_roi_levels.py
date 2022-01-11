# import tensorflow.compat.v1 as tf
import tensorflow as tf
import time
import numpy as np

tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


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
    scale = tf.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = tf.floor(tf.math.log(scale / finest_scale + 1e-6) / tf.math.log(2.))
    target_lvls = tf.clip_by_value(target_lvls, clip_value_min=0, clip_value_max=num_levels - 1)

    return target_lvls

@tf.function(experimental_compile=True)
def fast_map_roi_levels(rois, num_levels=4, finest_scale=56):
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
    scale = tf.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = tf.floor(tf.math.log(scale / finest_scale + 1e-6) / tf.math.log(2.))
    target_lvls = tf.clip_by_value(target_lvls, clip_value_min=0, clip_value_max=num_levels - 1)

    return target_lvls


def main():
    # assert error  结果不一致
    # M = 20
    # N = 5
    # g = tf.random.Generator.from_seed(0)
    # rois = g.normal(shape=[M, N], dtype=tf.float32)

    # 使用 rois_list.pth 中的数据   (20, 5)
    rois = tf.convert_to_tensor(
        [
        [0.0000000e+00, 3.6045001e+02, 9.5453171e+01, 8.7710858e+02, 5.1949365e+02],
        [0.0000000e+00, 1.8994501e+01, 1.3780254e+02, 3.5415225e+02, 4.1597974e+02],
        [0.0000000e+00, 1.5918526e+02, 7.5503799e+01, 2.4869026e+02, 2.8040506e+02],
        [0.0000000e+00, 8.7716931e+02, 9.7073418e+01, 1.0497196e+03, 5.5191901e+02],
        [0.0000000e+00, 2.4852827e+02, 1.3496709e+02, 3.0640277e+02, 1.9969620e+02],
        [0.0000000e+00, 4.8194092e+02, 0.0000000e+00, 9.7233264e+02, 5.1464478e+02],
        [0.0000000e+00, 1.6378830e+02, 1.5231029e+02, 9.6050061e+02, 5.0291547e+02],
        [0.0000000e+00, 2.4458139e+02, 1.5264964e+02, 3.0362997e+02, 2.0991206e+02],
        [1.0000000e+00, 8.9350793e+02, 4.2961905e+02, 1.1423363e+03, 5.4114288e+02],
        [1.0000000e+00, 6.0727148e+02, 4.3106665e+02, 7.0231543e+02, 4.6634286e+02],
        [1.0000000e+00, 3.6585239e+02, 5.3674286e+02, 5.0091376e+02, 5.7045715e+02],
        [1.0000000e+00, 4.8493343e+01, 4.5000000e+02, 1.3367097e+02, 4.7123810e+02],
        [1.0000000e+00, 5.1615125e+02, 2.2142857e+02, 6.6700250e+02, 4.4548572e+02],
        [1.0000000e+00, 4.6381046e+02, 2.7034283e+02, 7.4686609e+02, 4.2630475e+02],
        [1.0000000e+00, 4.2512625e+02, 3.8870477e+02, 4.6664844e+02, 4.3249524e+02],
        [1.0000000e+00, 2.2869583e+02, 4.2259048e+02, 2.3705740e+02, 4.3782858e+02],
        [1.0000000e+00, 4.2718332e+02, 4.9095239e+02, 4.6038202e+02, 5.4499048e+02],
        [1.0000000e+00, 2.2595308e+02, 4.2718097e+02, 3.5272910e+02, 4.5750476e+02],
        [1.0000000e+00, 2.5059975e+02, 4.1998096e+02, 3.8341360e+02, 4.4843808e+02],
        [1.0000000e+00, 2.2642924e+02, 2.4546666e+02, 3.8388977e+02, 4.4822858e+02]]
    )
    num_levels = 5
    finest_scale = 56
    sargs = [rois, num_levels, finest_scale]

    a = fast_map_roi_levels(*sargs)
    b = map_roi_levels(*sargs)
    assert np.allclose(a.numpy(), b.numpy())

    time_begin = time.time()
    for _ in range(10000):
        a = map_roi_levels(*sargs)
    _ = a.numpy()
    time_mid = time.time()
    for _ in range(10000):
        b = fast_map_roi_levels(*sargs)
    _ = b.numpy()
    time_end = time.time()

    print("time tensorflow: " + str(time_mid - time_begin))
    print("time xla: " + str(time_end - time_mid))


if __name__ == "__main__":
    main()