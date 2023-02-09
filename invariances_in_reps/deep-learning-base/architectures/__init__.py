import torch
import torch.nn as nn
from pytorch_lightning import utilities as pl_utils
from typing import Callable
from collections import OrderedDict
import logging, os

from . import cifar_models
### all paths are added in __init__.py of the top level dir which allows for the following import
from datasets import dataset_metadata as ds
import timm
import timm.models as models

_logger = logging.getLogger(__name__)

def list_models(dataset_name):
    if 'cifar' in dataset_name:
        return cifar_models.model_names
    else:
        return models.list_models()

def identity(x):
    return x

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict.copy()
            new_state_dict = OrderedDict()
            if sum([k.startswith('model') for k in state_dict.keys()]) > 0:
                for k, v in state_dict.items():
                    # strip `model.` prefix
                    if k.startswith('model'):
                        new_state_dict[k[6:]] = v
                state_dict = new_state_dict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def load_pretrained_weights(model, pretrained, checkpoint_path):
    if pretrained:
        assert checkpoint_path, 'Must pass checkpoint_path for pretrained CIFAR models'
        load_checkpoint(model, checkpoint_path) # from timm


def create_model(model_name: str, 
                dataset_name: str, 
                pretrained: bool = False,
                checkpoint_path: str = '',
                seed: int = 0, 
                loading_function: Callable = load_pretrained_weights, 
                callback: Callable = identity) -> nn.Module:
    """
    callback: function that can be used to alter the model after creation 
            (eg: adding more classification layers)
    loading_function: user-defined function that defines the logic for loading weights
            from a user-defined checkpoint file. Must take 3 args: model, pretrained, checkpoint_path
    """
    pl_utils.seed.seed_everything(seed)
    
    if dataset_name not in ds.DATASET_PARAMS:    dataset_name = 'imagenet'

    if 'cifar' in dataset_name:
        assert model_name in cifar_models.model_names, f'{model_name} not available for {dataset_name}'
        model = cifar_models.create_model_fn(model_name)(num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'])
        loading_function(model, pretrained, checkpoint_path)
    else:
        # Use timm for ImageNet and other big dataset models 
        should_custom_load = pretrained and checkpoint_path
        model = timm.create_model(model_name, 
                                  num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'],
                                  pretrained=False, # default loading happens via loading_function
                                  checkpoint_path='') # default loading happens via loading_function
        if should_custom_load:    loading_function(model, pretrained, checkpoint_path)

    return callback(model)