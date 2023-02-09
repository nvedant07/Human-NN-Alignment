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
parser.add_argument('--seed', type=int, default=420)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--simclr_augs', type=int, default=0)
args = parser.parse_args()


model = args.model
dataset = args.dataset
seed = args.seed
max_epochs = DATASET_PARAMS[dataset]['epochs'] if args.max_epochs is None else args.max_epochs
devices = 1

m1 = arch.create_model(model, dataset, pretrained=False, 
                       checkpoint_path='', seed=seed, 
                       callback=partial(LightningWrapper, dataset_name=dataset))
dm = DATA_MODULES[dataset](
    data_dir=<path>,
    transform_train=SIMCLR_TRAIN_TRANSFORMS(DATASET_PARAMS[dataset]['input_size']) \
        if bool(args.simclr_augs) else DATASET_PARAMS[dataset]['transform_train'])

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(
    dirpath='./checkpoints/'
            f'{dataset}/{model}_simclr_augs_{bool(args.simclr_augs)}', 
    filename='{epoch}', 
    every_n_epochs=50, 
    save_top_k=1, 
    save_last=False,
    verbose=True,
    mode='min', 
    monitor='val_loss')
## always use ddp for multi-GPU training -- works much faster, does not split batches
## can pass any quantity to LitProgressBar to be
## monitored during training, must be logged by the LightningModule 
## in `train_step_end` for it to be displayed during training
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=max_epochs,
                  check_val_every_n_epoch=1,
                  callbacks=[LitProgressBar(['running_acc']), 
                             checkpointer])
trainer.fit(m1, datamodule=dm)
