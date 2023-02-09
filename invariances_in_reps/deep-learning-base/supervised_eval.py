from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from training import NicerModelCheckpointing, LitProgressBar
import architectures as arch
from architectures.callbacks import LightningWrapper
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS, SIMCLR_TRAIN_TRANSFORMS
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--checkpoint_path', type=str, default='')
args = parser.parse_args()


model = args.model
dataset = args.dataset
devices = 1

m1 = arch.create_model(model, dataset, pretrained=True, 
                       checkpoint_path=args.checkpoint_path, 
                       callback=partial(LightningWrapper, dataset_name=dataset))
dm = DATA_MODULES[dataset](data_dir=<path>)

# pl_utils.seed.seed_everything(seed, workers=True)

## always use ddp for multi-GPU training -- works much faster, does not split batches
## can pass any quantity to LitProgressBar to be
## monitored during training, must be logged by the LightningModule 
## in `train_step_end` for it to be displayed during training
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=1,
                  check_val_every_n_epoch=1,
                  callbacks=[LitProgressBar(['running_acc'])])
trainer.test(m1, datamodule=dm)
