import numpy as np
import torch as ch
from torch.nn import functional as F
from human_nn_alignment.kornia_functions import translate, get_rotation_matrix2d, warp_affine


def _roundup(value):
    return np.ceil(value).astype(int)

def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle

def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner

def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        t = ch.tensor([[dx, dy]] * image_t.shape[0], device=image_t.device, dtype=ch.float)
        return translate(image_t, t)

    return inner


def pad(w, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, [w] * 4, mode=mode, value=constant_value,)

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [_roundup(scale * d) for d in shp]
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = ch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner

def random_rotate(angles, units="degrees"):
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = ch.ones(b, device=image_t.device) * alpha
        scale = ch.ones(b, 2, device=image_t.device)
        center = ch.ones(b, 2, device=image_t.device)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = get_rotation_matrix2d(center, angle, scale)
        rotated_image = warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner
