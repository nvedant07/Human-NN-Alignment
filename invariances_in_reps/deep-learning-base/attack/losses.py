from typing import Callable, Optional, Tuple, Union, List
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import os

import lpips

from architectures.inference import inference_with_features

ROBUST_MODEL_WEIGHTS = {
    'resnet18_madry_imagenet': '<path>',
    'resnet50_madry_imagenet': '<path>',
    'vgg16_bn_madry_imagenet': '<path>'
}


class BaseLoss:

    def __init__(self, *args, **kwargs):
        pass

    def _set_target_inps(self, target_inp) -> None:
        ## implement as necessary in derived classes
        pass

    def _set_transforms(self, t) -> None:
        self.transforms = t

    def _set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer
    
    def _set_fft(self, fft_transform) -> None:
        self.fft_transform = fft_transform
    
    def _transform_input(self, inp):
        ## CAUTION: Normalization must be applied **after** the transforms
        ## So pass the should_normalize flasg as False in attack_module and
        ## pass normalizer from attack_module to __init__ of BaseLoss
        if hasattr(self, 'fft_transform'):
            inp = self.fft_transform(inp)
        if isinstance(self, LPNormLossSingleModel) or \
            isinstance(self, LpNormLossSingleModelPerceptual):
            if hasattr(self, 'transforms'):
                ## transforms should only be applied for Losses 
                ## that operate on hidden reps
                inp = self.transforms(inp)
            if hasattr(self, 'normalizer'):
                inp = self.normalizer(inp)
        return inp

    def clear_cache(self) -> None:
        pass

    def __call__(self, model1, model2, inp, targ1, targ2) -> ch.Tensor:
        raise NotImplementedError('Call must be implemented in inherited class')


class LPNormLossSingleModel(BaseLoss):
    ## Regularizer-free
    def __init__(self, lpnorm_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpnorm_type = lpnorm_type

    def __call__(self, model1, model2, inp, targ1, targ2):
        inp = self._transform_input(inp)
        _, rep1 = inference_with_features(model1, inp, with_latent=True, fake_relu=True)
        self.model1_loss_normed = ch.div(ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1), 
                                         ch.norm(targ1, p=self.lpnorm_type, dim=1))
        self.model1_loss = ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1)
        loss = self.model1_loss_normed

        rep1 = None
        ch.cuda.empty_cache()

        return loss
    
    def clear_cache(self) -> None:
        self.model1_loss, self.model1_loss_normed = None, None
        ch.cuda.empty_cache()

    def __repr__(self):
        return f'Model1 Loss: {ch.mean(self.model1_loss)} ({ch.mean(self.model1_loss_normed)})'


class TVLoss(BaseLoss):
    ## CAUTION: Regularization must be applied to the original image
    ## So pass the should_normalize flasg as False in attack_module and
    ## pass normalizer from attack_module to __init__ of TVLoss

    # TV loss to penalize high freq features, see https://arxiv.org/abs/1412.0035
    def __init__(self, beta=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def __call__(self, model1, model2, inp, targ1, targ2):
        ## see kornia or tf's implemetation: 
        #   https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html
        #   https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/ops/image_ops_impl.py#L3220-L3289
        
        inp = self._transform_input(inp)

        if not isinstance(inp, ch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

        if len(inp.shape) != 4:
            raise ValueError("Expected input tensor to be of ndim 4 [batch, channels, height, width], "
                             f"but got ndim = {len(inp.shape)} instead.")

        pixel_dif1 = inp[..., 1:, :] - inp[..., :-1, :] # diff accross y
        pixel_dif2 = inp[..., :, 1:] - inp[..., :, :-1] # diff accross x

        reduce_axes = (-3, -2, -1)
        res1 = (pixel_dif1 ** 2).sum(dim=reduce_axes)
        res2 = (pixel_dif2 ** 2).sum(dim=reduce_axes)
        # res1 = pixel_dif1.abs().sum(dim=reduce_axes)
        # res2 = pixel_dif2.abs().sum(dim=reduce_axes)
        self.tv_dist = ch.pow(res1 + res2, self.beta/2.)
        return self.tv_dist
    
    def clear_cache(self) -> None:
        self.tv_dist = None
        ch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f'TVDist: {ch.mean(self.tv_dist) if hasattr(self, "tv_dist") else 0.}'


class L1Loss(BaseLoss):
    ## CAUTION: Regularization must be applied to the original image
    ## So pass the should_normalize flasg as False in attack_module and
    ## pass normalizer from attack_module to __init__ of L1Loss
    # L1 norm of input -- sparsity constraint
    def __init__(self, constant=.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def __call__(self, model1, model2, inp, targ1, targ2) -> ch.Tensor:
        inp = self._transform_input(inp)
        self.loss = ch.sum(ch.abs(inp - self.constant), 
                           dim=list(range(1,len(inp.shape))))
        return self.loss

    def clear_cache(self) -> None:
        self.loss = None
        ch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f'L1Loss: {ch.mean(self.loss) if hasattr(self, "loss") else 0.}'


class LpLoss(BaseLoss):
    ## CAUTION: Regularization must be applied to the original image
    ## So pass the should_normalize flasg as False in attack_module and
    ## pass normalizer from attack_module to __init__ of L1Loss
    # Lp norm of flattened input
    def __init__(self, p=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def __call__(self, model1, model2, inp, targ1, targ2) -> ch.Tensor:
        inp = self._transform_input(inp)
        self.loss = ch.linalg.norm(inp.flatten(1), dim=1, ord=self.p)
        return self.loss

    def clear_cache(self) -> None:
        self.loss = None
        ch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f'L{self.p}Loss: {ch.mean(self.loss) if hasattr(self, "loss") else 0.}'


class BlurLoss(BaseLoss):
    ## CAUTION: Regularization must be applied to the original image
    ## So pass the should_normalize flasg as False in attack_module and
    ## pass normalizer from attack_module to __init__ of BlurLoss

    # Pushes input towards a blurred version
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _blur(self, x, w=3):
        depth = x.shape[1] # channels
        k = ch.zeros([depth, depth, w, w], device=x.device)
        for c in range(depth):
            k_ch = k[c, c, :, :]
            k_ch[:,:] = 0.5
            k_ch[1:-1, 1:-1] = 1.0

        conv_k = lambda t: F.conv2d(t, weight=k, stride=[1, 1], padding="same")
        return conv_k(x) / conv_k(ch.ones_like(x))

    def __call__(self, model1, model2, inp, targ1, targ2) -> ch.Tensor:
        inp = self._transform_input(inp)
        
        with ch.no_grad():
            blurred_inp = self._blur(inp)
        
        self.loss = 0.5*ch.sum((inp - blurred_inp)**2, 
                                dim=list(range(1,len(inp.shape))))
        return self.loss
    
    def clear_cache(self) -> None:
        self.loss = None
        ch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return f'BlurLoss: {ch.mean(self.loss) if hasattr(self, "loss") else 0.}'


class LpNormLossSingleModelPerceptual(BaseLoss):
    """
    Uses perceptual loss (LPIPS) to simulate human perception, goal of this loss 
    is to generate images which are perceptually different from the target.
    """
    def __init__(self, lpips_model, lpips_model_path, 
                 lpnorm_type, scaling_factor=None,
                 *args, **kwargs):
        """
        lpips_model: model architecture
            alex (for finetuned weights, use 
                   PerceptualSimilarity/lpips/weights/v0.1/alex.pth,
                  for ImageNet weights, use pretrained=False)
            vgg16 (for finetuned weights, use
                   PerceptualSimilarity/lpips/weights/v0.1/vgg.pth,
                   for ImageNet weights, use pretrained=False)
            squeeze (for finetuned weights, use 
                   PerceptualSimilarity/lpips/weights/v0.1/squeeze.pth,
                   for ImageNet weights, use pretrained=False)
            resnet18
            resnet18_madry_imagenet
            resnet50_madry_imagenet
            vgg16_bn_madry_imagenet

        lpips_model_path: if set to None, then standard ImageNet weights are used. 
            For models of the form *_madry_*, there must be a path, if None is specified, 
            then the default ones are used (defined in `ROBUST_MODEL_WEIGHTS`)
        """
        super().__init__(*args, **kwargs)
        self.lpnorm_type = lpnorm_type
        if lpips_model in ROBUST_MODEL_WEIGHTS and lpips_model_path is None:
            model_path = ROBUST_MODEL_WEIGHTS[lpips_model]
        else:
            model_path = lpips_model_path
        self.lpips = lpips.LPIPS(net=lpips_model, 
                                 model_path=model_path, 
                                 pretrained=model_path is not None)
        self.target_inp = None
        self.scaling_factor = scaling_factor

    def _set_target_inps(self, target_inp) -> None:
        self.target_inp = target_inp

    def __call__(self, model1, model2, inp, targ1, targ2):
        self.lpips = self.lpips.to(inp.device)
        assert self.target_inp is not None, 'Must call `set_target_inps` first!'

        _, rep1 = inference_with_features(model1, inp, with_latent=True, fake_relu=True)
        self.model1_loss_normed = ch.div(ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1), 
                                            ch.norm(targ1, p=self.lpnorm_type, dim=1))
        self.model1_loss = ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1)
        # LPIPS expects images in [-1, 1] so pass normalize = True
        self.lpips_distance = self.lpips(inp, self.target_inp, normalize=True).squeeze()

        if self.scaling_factor is None:
            # set it initially, otherwise this will never converge
            self.scaling_factor = ch.mean(self.model1_loss_normed).item()/ch.mean(self.lpips_distance).item()
        self.loss = self.model1_loss_normed - self.scaling_factor * self.lpips_distance

        rep1 = None
        ch.cuda.empty_cache()

        return self.loss

    def clear_cache(self) -> None:
        self.model1_loss_normed, self.model1_loss, self.lpips_distance = None, None, None
        self.loss, self.lpips_distance = None, None
        self.lpips = self.lpips.cpu()
        ch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f'Loss: {ch.mean(self.loss)}, Model1 Loss: {ch.mean(self.model1_loss)} ({ch.mean(self.model1_loss_normed)})'\
               f'LPIPS Dist: {ch.mean(self.lpips_distance)}'


class CompositeLoss(BaseLoss):
    """
    Combines multiple losses. Eg:

    CompositeLoss(losses=(LpNormLossSingleModel(lpnorm_type=2), TVLoss(lpnorm_type=2)), 
                  weights=(1., 1.))
    would optimize for close representations
    """
    def __init__(self, losses: Union[List[BaseLoss], Tuple[BaseLoss]], 
                       weights: Optional[Union[ch.Tensor, list, tuple, np.ndarray]]=None,
                       *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights
        if weights is None:
            self.weights = np.ones(len(losses))
        self.losses = losses

    def _set_transforms(self, t) -> None:
        self.transforms = t
        for l in self.losses:
            l._set_transforms(t)
    
    def _set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer
        for l in self.losses:
            l._set_normalizer(normalizer)
    
    def _set_fft(self, fft_transform) -> None:
        self.fft_transform = fft_transform
        for l in self.losses:
            l._set_fft(fft_transform)

    def __call__(self, model1, model2, inp, targ1, targ2):
        total_loss = 0.
        for i,l in enumerate(self.losses):
            total_loss += self.weights[i] * l(model1, model2, inp, targ1, targ2)
        return total_loss
    
    def clear_cache(self) -> None:
        for l in self.losses:
            l.clear_cache()

    def __repr__(self) -> str:
        return ' '.join([str(l) for l in self.losses])

