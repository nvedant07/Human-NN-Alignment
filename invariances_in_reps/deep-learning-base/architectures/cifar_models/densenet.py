'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_modules import FakeReLU
from .registry import register_model_name


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, num_layers=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        if num_layers > 1:
            fc_layers = []
            for _ in range(num_layers - 1):    fc_layers.append(nn.Linear(num_planes, num_planes))
            fc_layers.append(nn.Linear(num_planes, num_classes))
            self.linear = nn.Sequential(*fc_layers)
        else:
            self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, layer_num=None):
        assert not no_relu, \
            "DenseNet has no pre-ReLU activations, no_relu not supported"
        layer_count = 0
        out = self.conv1(x)
        
        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()
        
        out = self.dense1(out)
        
        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()
        
        out = self.trans1(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        out = self.dense2(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        out = self.trans2(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        out = self.dense3(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        out = self.trans3(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        out = self.dense4(out)

        layer_count += 1
        if layer_count == layer_num:
            latent = out.clone()

        if fake_relu:
            out = F.avg_pool2d(FakeReLU.apply(self.bn(out)), 4)
        else:
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        
        layer_count += 1
        if with_latent or layer_count == layer_num:    latent = out.clone()
        
        out = self.linear(out)
        
        if with_latent:
            return out, latent
        if layer_num is not None:
            assert not with_latent
            return out, latent
        return out

@register_model_name
def densenet121(**kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, **kwargs)

@register_model_name
def densenet161(**kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, **kwargs)

@register_model_name
def densenet169(**kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, **kwargs)

@register_model_name
def densenet201(**kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, **kwargs)
