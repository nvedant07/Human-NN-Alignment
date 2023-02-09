## functions taken from kornia.geometry for consistent re-use
## See: https://github.com/kornia/kornia/blob/master/kornia/geometry/transform/affwarp.py

import torch
from torch.nn import functional as F
from typing import Tuple, Optional

def translate(
    tensor: torch.Tensor,
    translation: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Translate the tensor in pixel units.
    .. image:: _static/img/translate.png
    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
    Returns:
        The translated tensor with shape as input.
    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = translate(img, translation)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(translation, torch.Tensor):
        raise TypeError(f"Input translation type is not a torch.Tensor. Got {type(translation)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], mode, padding_mode, align_corners)

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Compute affine matrix for translation."""
    matrix: torch.Tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix

def affine(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> torch.Tensor:
    r"""Apply an affine transformation to the image.
    .. image:: _static/img/warp_affine.png
    Args:
        tensor: The image tensor to be warped in shapes of
            :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
        matrix: The 2x3 affine transformation matrix.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
    Returns:
        The warped image with the same shape as the input.
    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> aff = torch.eye(2, 3)[None]
        >>> out = affine(img, aff)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
    fill_value: torch.Tensor = torch.zeros(3),  # needed for jit
) -> torch.Tensor:
    r"""Apply an affine transformation to a tensor.
    .. image:: _static/img/warp_affine.png
    The function warp_affine transforms the source tensor using
    the specified matrix:
    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )
    Args:
        src: input tensor of shape :math:`(B, C, H, W)`.
        M: affine transformation of shape :math:`(B, 2, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        align_corners : mode for grid_generation.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.
    Returns:
        the warped tensor with shape :math:`(B, C, H, W)`.
    .. note::
        This function is often used in conjunction with :func:`get_rotation_matrix2d`,
        :func:`get_shear_matrix2d`, :func:`get_affine_matrix2d`, :func:`invert_affine_transform`.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       rotate_affine.html>`__.
    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> A = torch.eye(2, 3)[None]
       >>> out = warp_affine(img, A, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError(f"Input M must be a Bx2x3 tensor. Got {M.shape}")

    # fill padding is only supported for 3 channels because we can't set fill_value default
    # to None as this gives jit issues.
    if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
        raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=align_corners)

    if padding_mode == "fill":
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def _fill_and_warp(
    src: torch.Tensor, grid: torch.Tensor, mode: str, align_corners: bool, fill_value: torch.Tensor
) -> torch.Tensor:
    r"""Warp a mask of ones, then multiple with fill_value and add to default warp.
    Args:
        src: input tensor of shape :math:`(B, 3, H, W)`.
        grid: grid tensor from `transform_points`.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.
    Returns:
        the warped and filled tensor with shape :math:`(B, 3, H, W)`.
    """
    ones_mask = torch.ones_like(src)
    fill_value = fill_value.to(ones_mask)[None, :, None, None]  # cast and add dimensions for broadcasting
    inv_ones_mask = 1 - F.grid_sample(ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros")
    inv_color_mask = inv_ones_mask * fill_value
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask

def convert_affinematrix_to_homography(A: torch.Tensor) -> torch.Tensor:
    r"""Function that converts batch of affine matrices.
    Args:
        A: the affine matrix with shape :math:`(B,2,3)`.
    Returns:
         the homography matrix with shape of :math:`(B,3,3)`.
    Examples:
        >>> A = torch.tensor([[[1., 0., 0.],
        ...                    [0., 1., 0.]]])
        >>> convert_affinematrix_to_homography(A)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)

def _convert_affinematrix_to_homography_impl(A: torch.Tensor) -> torch.Tensor:
    H: torch.Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H

def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """
    NOTE: inverse must happen on CPU esle this throws an error
    Helper function to make torch.inverse work with other than fp32/64.
    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    device = input.device
    return torch.inverse(input.to(dtype=dtype, device='cpu')).to(dtype=input.dtype, device=device)


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors
    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.
    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)

def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    r"""Calculate an affine matrix of 2D rotation.
    The function calculates the following matrix:
    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}
    where
    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})
    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.
    Args:
        center: center of the rotation in the source image with shape :math:`(B, 2)`.
        angle: rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner) with shape :math:`(B)`.
        scale: scale factor for x, y scaling with shape :math:`(B, 2)`.
    Returns:
        the affine matrix of 2D rotation with shape :math:`(B, 2, 3)`.
    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones((1, 2))
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    .. note::
        This function is often used in conjunction with :func:`warp_affine`.
    """
    if not isinstance(center, torch.Tensor):
        raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

    if not isinstance(angle, torch.Tensor):
        raise TypeError(f"Input angle type is not a torch.Tensor. Got {type(angle)}")

    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"Input scale type is not a torch.Tensor. Got {type(scale)}")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 tensor. Got {center.shape}")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B tensor. Got {angle.shape}")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a Bx2 tensor. Got {scale.shape}")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Got center {}, angle {} and scale {}".format(
                center.shape, angle.shape, scale.shape
            )
        )

    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            "Inputs must have same device Got center ({}, {}), angle ({}, {}) and scale ({}, {})".format(
                center.device, center.dtype, angle.device, angle.dtype, scale.device, scale.dtype
            )
        )

    shift_m = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3

def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    r"""Create a rotation matrix out of angles in degrees.
    Args:
        angle: tensor of angles in degrees, any shape :math:`(*)`.
    Returns:
        tensor of rotation matrices with shape :math:`(*, 2, 2)`.
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)

def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians.
    Args:
        tensor: Tensor of arbitrary shape.
    Returns:
        tensor with same shape as input.
    Examples:
        >>> input = torch.tensor(180.)
        >>> deg2rad(input)
        tensor(3.1416)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")
    pi = torch.tensor(3.14159265358979323846, device=tensor.device, dtype=tensor.dtype)
    return tensor * pi / 180.0