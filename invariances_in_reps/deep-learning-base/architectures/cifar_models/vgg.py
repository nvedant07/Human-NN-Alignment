'''VGG11/13/16/19 in Pytorch.'''
from .registry import register_model_name
import torch
import torch.nn as nn
from typing import Iterator
import warnings

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, num_layers=1):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        self.num_fc_layers = num_layers
        if num_layers > 1:
            in_ftrs = 512
            fc_layers = []
            for _ in range(num_layers - 1):    fc_layers.append(nn.Linear(in_ftrs, in_ftrs))
            fc_layers.append(nn.Linear(in_ftrs, num_classes))
            self.classifier = nn.Sequential(*fc_layers)
        else:
            self.classifier = nn.Linear(512, num_classes)
    
    def num_layers(self) -> Iterator:
        return range(1, len(self.features) + self.num_fc_layers + 1)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, layer_num=None):
        if (not fake_relu) and (not no_relu):
            warnings.warn("`fake_relu` and `no_relu` not yet supported for this architecture; "
                          "there will be no effect on the forward function")
        if layer_num is not None:
            ft_1, ft_2 = [], []
            for idx, l in enumerate(self.features):
                if idx + 1 <= layer_num:
                    ft_1.append(l)
                else:
                    ft_2.append(l)
            m1 = nn.Sequential(*ft_1)
            m2 = nn.Sequential(*ft_2)
            latent = m1(x)
            out = m2(latent)
            out = out.view(out.size(0), -1)
            if len(ft_2) == 0: # penultimate layer
                latent = out.clone()
        else:
            out = self.features(x)
            out = out.view(out.size(0), -1)
        
        if with_latent:
            latent = out.clone()
        
        out = self.classifier(out)
        
        if with_latent:
            return out, latent
        if layer_num is not None:
            assert not with_latent
            return out, latent
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

@register_model_name
def vgg11(**kwargs):
    return VGG('VGG11', **kwargs)

@register_model_name
def vgg13(**kwargs):
    return VGG('VGG13', **kwargs)

@register_model_name
def vgg16(**kwargs):
    return VGG('VGG16', **kwargs)

@register_model_name
def vgg19(**kwargs):
    return VGG('VGG19', **kwargs)
