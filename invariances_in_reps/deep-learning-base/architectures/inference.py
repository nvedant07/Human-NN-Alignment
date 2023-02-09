from time import pthread_getcpuclockid
import warnings
import torch
import torch.nn as nn
from timm.models import layers
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from .cifar_models.custom_modules import FakeReLUM


def check_fake_and_no_relu(model, args, kwargs):
    if model.__class__.__name__ == 'ResNet':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            for block in model.layer4:
                block.__setattr__('act3', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) in kwargs or (len(args) > 2 and args[2]):
            for block in model.layer4:
                block.__setattr__('act3', nn.Identity())
    elif model.__class__.__name__ == 'DenseNet':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            model.features.norm5.__setattr__('act', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) or (len(args) > 2 and args[2]):
            warnings.warn('no_relu not applicable for DenseNet, this will have no effect')
    elif model.__class__.__name__ == 'VGG':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            model.pre_logits.__setattr__('act2', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) or (len(args) > 2 and args[2]):
            model.pre_logits.__setattr__('act2', nn.Identity())


def inference_with_features(model: nn.Module, 
                            X: torch.Tensor, *args, **kwargs):
    if (('with_latent' in kwargs and kwargs['with_latent']) or \
        (len(args) and args[0])) and \
            hasattr(model, 'forward_features'):
            
            check_fake_and_no_relu(model, args, kwargs)

            out = model.forward_features(X)
            if model.__class__.__name__ == 'VGG': 
                # special case; pooling is done in the head
                out = model.forward_head(out, pre_logits=True)
                x_latent = model.head(out, pre_logits=True)
            else:
                # these features need to be pooled
                if hasattr(model, 'global_pool'):
                    out = model.global_pool(out) 
                    # check if global_pool contains Flatten
                    if not isinstance(model.global_pool.flatten, nn.Flatten):
                        out = nn.Flatten(1)(out)
                else:
                    pooling = layers.SelectAdaptivePool2d(pool_type='avg', 
                                                        flatten=True)
                    out = pooling(out)
                x_latent = out
            
            possible_names = ['fc', 'head']
            for name in possible_names:
                if hasattr(model, name):
                    classifer = model.__getattr__(name)
                    break
            try:
                pred = classifer(out)
            except:
                warnings.warn(f'Model classifier not in {possible_names}, '
                              'doing another forward pass')
                pred = model(X)

            return pred, x_latent
    
    return model(X, *args, **kwargs)
