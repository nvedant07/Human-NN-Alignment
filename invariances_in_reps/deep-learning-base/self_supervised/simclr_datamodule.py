from datasets.dataset_metadata import SIMCLR_TRAIN_TRANSFORMS, SIMCLR_TRAIN_TRANSFORMS_NOCOLOR
from .contrastive_data import ContrastiveTransformations


def simclr_dm(datamodule, s=1, views=2, nocolor=False):
    T = SIMCLR_TRAIN_TRANSFORMS_NOCOLOR if nocolor else SIMCLR_TRAIN_TRANSFORMS
    datamodule.__setattr__('transform_train', ContrastiveTransformations(
            T(datamodule.input_size, s), views))
    datamodule.setup() ## need to re-initialize train_ds

