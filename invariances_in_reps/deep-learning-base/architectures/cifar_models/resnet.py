'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_modules import SequentialWithArgs, FakeReLU
from . import models
from .registry import register_model_name
import torchvision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)

class SparseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=False):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_relu = use_relu
        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False):
        if fake_relu:
            raise ValueError('Not yet supported for Sparse Models!')
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=True):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU()

        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse3 = models.sparse_func_dict[sparse_func](sparsity)

        self.use_relu = use_relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False):
        if fake_relu:
            raise ValueError('Not yet supported for Sparse Models!')
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)

        out = self.bn2(self.conv2(out))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        if self.use_relu:
            out = self.relu(out)
        out = self.sparse3(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=10, use_relu=True, sparse_func='reg', bias=True):
        super(SparseResNet, self).__init__()
        self.in_planes = 64
        self.use_relu = use_relu


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0], sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1], sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2], sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3], sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu = nn.ReLU()

        self.activation = {}

    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sparsity, self.use_relu, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, layer_num=None):
        if fake_relu:
            raise ValueError('Not yet supported for Sparse Models!')
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        out = self.linear(pre_out)
        if with_latent:
            return out, pre_out
        return out


class SparseResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=1000, sparse_func='vol', bias=False):
        super(SparseResNet_ImageNet, self).__init__()
        self.in_planes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0], sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1], sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2], sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3], sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.sp = models.sparse_func_dict[sparse_func](sparsities[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.activation = {}

    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sparsity, use_relu=False, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, layer_num=None):
        if fake_relu:
            raise ValueError('Not yet supported for Sparse Models!')
        out = self.sp(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        pre_out = out.view(out.size(0), -1)
        out = self.linear(pre_out)
        if with_latent:
            return out, pre_out
        return out


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    # num_layers is for additional fully connected layers at the end
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, num_layers=1):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.num_fc_layers = num_layers
        if num_layers > 1:
            in_ftrs = feat_scale*widths[3]*block.expansion
            fc_layers = []
            for _ in range(num_layers - 1):    fc_layers.append(nn.Linear(in_ftrs, in_ftrs))
            fc_layers.append(nn.Linear(in_ftrs, num_classes))
            self.linear = nn.Sequential(*fc_layers)
        else:
            self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

    def num_layers(self) -> Iterator:
        return range(1, 5 + self.num_fc_layers + 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, layer_num=None):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4, fake_relu=fake_relu)
        pre_out = F.avg_pool2d(out5, 4)
        pre_out = pre_out.view(pre_out.size(0), -1)
        # return pre_out
        final = self.linear(pre_out)
        all_ops = [out1, out2, out3, out4, out5, pre_out]
        if with_latent:
            return final, pre_out
        if layer_num is not None:
            assert not with_latent and layer_num <= len(all_ops)
            return final, all_ops[layer_num - 1]
        return final

@register_model_name
def resnet18(**kwargs):
    # return torchvision.models.resnet18(pretrained=False, num_classes=10)
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

@register_model_name
def resnet18wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=5, **kwargs)

@register_model_name
def resnet18thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=.75, **kwargs)

@register_model_name
def resnet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

@register_model_name
def resnet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

@register_model_name
def resnet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

@register_model_name
def resnet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

@register_model_name
def sparseresnet18(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False, **kwargs):
    return SparseResNet(SparseBasicBlock, [2,2,2,2], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

@register_model_name
def sparseresnet34(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False, **kwargs):
    return SparseResNet(SparseBasicBlock, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

@register_model_name
def sparseresnet50(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False, **kwargs):
    return SparseResNet(SparseBottleneck, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

@register_model_name
def sparseresnet101(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False, **kwargs):
    return SparseResNet(SparseBottleneck, [3,4,23,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

@register_model_name
def sparseresnet152(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False, **kwargs):
    return SparseResNet(SparseBottleneck, [3,8,36,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)
