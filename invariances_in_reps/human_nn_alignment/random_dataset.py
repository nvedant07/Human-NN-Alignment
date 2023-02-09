import torch
from typing import Optional

from datasets.data_modules import BaseDataModule
from torch.utils.data import Dataset

class RandomDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(RandomDataset, *args, **kwargs)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_ds = self.dataset_class(
            **self.dataset_kwargs)
        self.val_ds = self.train_ds
        self.test_ds = self.train_ds


class RandomDataset(Dataset):
    """
    Generates random images
    """

    def __init__(self, num_samples, mean=1., std=0., shape=(3,32,32)):
        self.mean = mean
        self.std = std
        self.shape = shape
        self.num_samples = num_samples
    
    def __getitem__(self, index):
        label = 0
        img = torch.normal(mean=self.mean * torch.ones(self.shape), 
                           std=self.std * torch.ones(self.shape))
        return img, label

    def __len__(self):
        return self.num_samples
