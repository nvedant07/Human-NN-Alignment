import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset, Sampler, Subset
import pytorch_lightning as pl
from pytorch_lightning import utilities as pl_utils
from typing import Dict, Iterable, Optional, Sequence, Union, Callable
from .dataset_metadata import DATASET_PARAMS
import numpy as np


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_class: Callable[[], Dataset],
                 data_dir: str = '/tmp/', 
                 shuffle_train: bool = True, 
                 shuffle_val: bool = False, 
                 shuffle_test: bool = False, 
                 pin_memory: bool = True, 
                 workers: int = 30,
                 random_split: int = 0, 
                 subset: Optional[float] = None,
                 subset_type: str = 'rand',
                 batch_size: Optional[int] = None, 
                 batch_sampler: Optional[Union[Sampler[Sequence], Iterable[Sequence]]] = None, 
                 transform_train: Optional[Callable] = None, 
                 transform_test: Optional[Callable] = None, 
                 val_frac: int = 0.1,
                 dataset_kwargs: Dict = {}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # if passing a custom dataset class, the params should be compatible with 
        # the setup() and prepare_data() functions in downstream data class
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.pin_memory = pin_memory
        self.batch_sampler = batch_sampler
        self.workers = workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.random_split = random_split
        self.val_frac = val_frac
        self.subset = subset
        self.subset_type = subset_type

    def train_dataloader(self):
        if not hasattr(self, 'train_ds'):
            self.setup()
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_train, 
            num_workers=self.workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        if not hasattr(self, 'val_ds'):
            self.setup()
        return DataLoader(self.val_ds if len(self.val_ds) > 0 else self.train_ds, 
            batch_size=self.batch_size, 
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_val, 
            num_workers=self.workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        if not hasattr(self, 'test_ds'):
            self.setup()
        return DataLoader(self.test_ds, batch_size=self.batch_size,             
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_test, 
            num_workers=self.workers, pin_memory=self.pin_memory)
    
    def subset_train_ds(self):
        train_sample_count = len(self.train_ds)
        if self.subset_type == 'rand':
            subset = np.random.choice(list(range(train_sample_count)), 
                size=self.subset, replace=False)
        elif self.subset_type == 'first':
            subset = np.arange(0, subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)
        self.train_ds = Subset(self.train_ds, subset)

    def init_remaining_attrs(self, dname):
        for k, v in DATASET_PARAMS[dname].items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.CIFAR10, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['cifar10']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['cifar10']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['cifar10']['batch_size']
        
        self.init_remaining_attrs('cifar10')

    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit', 'validate'):
            full_ds = self.dataset_class(root=self.data_dir, train=True, 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int((1-self.val_frac)*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
            if self.subset:
                self.subset_train_ds()
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, train=False, transform=self.transform_test, **self.dataset_kwargs)

class CIFAR100DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.CIFAR100, *args, **kwargs):
        super(CIFAR100DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['cifar100']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['cifar100']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['cifar100']['batch_size']
        
        self.init_remaining_attrs('cifar100')
    
    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit', 'validate'):
            full_ds = self.dataset_class(root=self.data_dir, train=True, 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int((1-self.val_frac)*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
            self.val_ds.__setattr__('transform', self.transform_test)
            if self.subset:
                self.subset_train_ds()
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, train=False, transform=self.transform_test, **self.dataset_kwargs)

class STL10DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.STL10, *args, **kwargs):
        super(STL10DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['stl10']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['stl10']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['stl10']['batch_size']
        
        self.init_remaining_attrs('stl10')
    
    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, split='train+unlabeled', download=True)
        self.dataset_class(self.data_dir, split='test', download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit', 'validate'):
            self.train_ds = self.dataset_class(root=self.data_dir, 
                split='train+unlabeled', transform=self.transform_train, 
                **self.dataset_kwargs)
            self.val_ds = self.dataset_class(root=self.data_dir, split='train', 
                transform=self.transform_test, **self.dataset_kwargs)
            if self.subset:
                self.subset_train_ds()
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, split='test',
                transform=self.transform_test, **self.dataset_kwargs)


class ImageNetDataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.ImageNet, *args, **kwargs):
        super(ImageNetDataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['imagenet']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['imagenet']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['imagenet']['batch_size']
        
        self.init_remaining_attrs('imagenet')
    
    def prepare_data(self):
        # ImageNet needs to be pre-downloaded; this step will unzip the directory
        self.dataset_class(root=self.data_dir, split='train')
        self.dataset_class(root=self.data_dir, split='val')

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit', 'validate'):
            full_ds = self.dataset_class(root=self.data_dir, split='train', 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int((1-self.val_frac)*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
            self.val_ds.__setattr__('transform', self.transform_test)
            if self.subset:
                self.subset_train_ds()
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, split='val', 
                transform=self.transform_test, **self.dataset_kwargs)


DATA_MODULES = {
    'imagenet': ImageNetDataModule, 
    'cifar10': CIFAR10DataModule, 
    'cifar100': CIFAR100DataModule, 
    'stl10': STL10DataModule
}