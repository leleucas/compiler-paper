import time
import torch

@torch.jit.script
def fast_tblr(priors,
                tblr,
                normalizer=torch.FloatTensor([4.0]).cuda(),
                normalize_by_wh=torch.IntTensor([1]).cuda(),
                max_shape=torch.IntTensor([0]).cuda()):
    '''if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)'''
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr
    top, bottom, left, right = loc_decode.split(1, dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if max_shape is not None:
        return boxes
        boxes[:, 0].clamp_(min=0, max=max_shape[1])
        boxes[:, 1].clamp_(min=0, max=max_shape[0])
        boxes[:, 2].clamp_(min=0, max=max_shape[1])
        boxes[:, 3].clamp_(min=0, max=max_shape[0])
    return boxes


def tblr2bboxes(priors,
                tblr,
                normalizer=torch.FloatTensor([4.0]).cuda(),
                normalize_by_wh=torch.IntTensor([1]).cuda(),
                max_shape=torch.IntTensor([0]).cuda()):
    """Decode tblr outputs to prediction boxes.

    The process includes 3 steps: 1) De-normalize tblr coordinates by
    multiplying it with `normalizer`; 2) De-normalize tblr coordinates by the
    prior bbox width and height if `normalize_by_wh` is `True`; 3) Convert
    tblr (top, bottom, left, right) pair relative to the center of priors back
    to (xmin, ymin, xmax, ymax) coordinate.

    Args:
        priors (Tensor): Prior boxes in point form (x0, y0, x1, y1)
          Shape: (n,4).
        tblr (Tensor): Coords of network output in tblr form
          Shape: (n, 4).
        normalizer (Sequence[float] | float): Normalization parameter of
          encoded boxes. By list, it represents the normalization factors at
          tblr dims. By float, it is the unified normalization factor at all
          dims. Default: 4.0
        normalize_by_wh (bool): Whether the tblr coordinates have been
          normalized by the side length (wh) of prior bboxes.
        max_shape (tuple, optional): Shape of the image. Decoded bboxes
          exceeding which will be clamped.

    Return:
        encoded boxes (Tensor), Shape: (n, 4)
    """
    '''if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)'''
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr
    top, bottom, left, right = loc_decode.split(1, dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if max_shape is not None:
        return boxes
        boxes[:, 0].clamp_(min=0, max=max_shape[1])
        boxes[:, 1].clamp_(min=0, max=max_shape[0])
        boxes[:, 2].clamp_(min=0, max=max_shape[1])
        boxes[:, 3].clamp_(min=0, max=max_shape[0])
    return boxes

if __name__ == "__main__":
    # data prepare
    N = 128
    shape = (N, 4)
    priors = torch.randn(shape)
    tblr = torch.randn(shape)
 
    priors = priors.cuda()
    tblr = tblr.cuda()

    sfunc = tblr2bboxes
    sargs = [priors, tblr]

    # correctness check
    assert torch.allclose(fast_tblr(*sargs), tblr2bboxes(*sargs), equal_nan=True)

    # performance check
    torch.cuda.synchronize()
    time_before = time.time()

    for _ in range(10000):
        ret = tblr2bboxes(*sargs)

    torch.cuda.synchronize()
    time_mid = time.time()

    for _ in range(10000):
        ret = fast_tblr(*sargs)

    torch.cuda.synchronize()
    time_end = time.time()
    print("time costing origin: " + str(time_mid - time_before))
    print("time costing coderized: " + str(time_end - time_mid))

    # get timeline
    '''parrots.runtime.profile(enable=True, file='profile.txt', use_scope=True)
    for _ in range(10):
        ret = sfunc._pyfunc(*sargs)

    for _ in range(10):
        ret = sfunc(*sargs)
    parrots.runtime.profile(enable=False)'''
