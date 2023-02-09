import numpy as np
import torch

TORCH_VERSION = torch.__version__


color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]


def _linear_decorrelate_color(tensor):
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(
        t_permute, 
        torch.tensor(color_correlation_normalized.T, device=tensor.device)
        )
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def to_valid_rgb(image_fft):
    image_fft = _linear_decorrelate_color(image_fft)
    return torch.sigmoid(image_fft)


def fft_image(og_shape, decay_power=1):
    """
    image is in the fourier domain
    """
    batch, channels, h, w = og_shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (channels,) + freqs.shape + (2,)
    
    def inner(image):
        assert image.shape[1:] == init_val_size
        scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
        scale = torch.tensor(scale, device=image.device).float()[None, None, ..., None]

        scaled_spectrum_t = scale * image
        if TORCH_VERSION >= "1.7.0":
            if type(image) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            _fft_image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            _fft_image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        _fft_image = _fft_image[:batch, :channels, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        _fft_image = _fft_image / magic
        return to_valid_rgb(_fft_image)

    return inner
