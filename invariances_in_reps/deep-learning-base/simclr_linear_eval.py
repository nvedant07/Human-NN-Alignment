# quick testing script
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torchvision.transforms import transforms
from training import NicerModelCheckpointing, LitProgressBar
import torch
import torch.nn as nn
import architectures as arch
from attack.callbacks import AdvCallback
from architectures.callbacks import LightningWrapper, AdvAttackWrapper, LinearEvalWrapper
from architectures.inference import inference_with_features
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from self_supervised.simclr_datamodule import simclr_dm
from self_supervised.simclr_callback import SimCLRWrapper
from functools import partial
import glob, os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--source_dataset', type=str, default='cifar10')
parser.add_argument('--target_dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--simclr_batch_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--adv_aug', type=bool, default=False)
args = parser.parse_args()



source_dataset = args.source_dataset
target_dataset = args.target_dataset
# dataset = 'imagenet'

model = args.model
# model = 'resnet50'

pretrained = True
seed = args.seed
devices = 1
num_nodes = 1
batch_size = args.batch_size if args.batch_size else \
    DATASET_PARAMS[target_dataset]['batch_size']
max_epochs = args.max_epochs if args.max_epochs else \
    DATASET_PARAMS[target_dataset]['epochs']

imagenet_path = <path>
data_path = <path>

dm = DATA_MODULES[target_dataset](
    data_dir=imagenet_path if target_dataset == 'imagenet' else data_path,
    val_frac=0.,
    batch_size=batch_size,
    transform_train=DATASET_PARAMS[source_dataset]['transform_train'],
    transform_test=DATASET_PARAMS[source_dataset]['transform_test'])


adv = f'{args.simclr_batch_size}_adv_{args.adv_aug}' if args.adv_aug else args.simclr_batch_size
dirpath = f'./checkpoints/{source_dataset}/{model}/simclr_bs_{adv}/lars_excludebn_True'

model_checkpoints = glob.glob(f'{dirpath}/*_rand_seed_{args.seed}.ckpt')

for checkpoint_path in model_checkpoints:
    if os.path.exists(f'{checkpoint_path.split(".ckpt")[0]}_{target_dataset}_linear_eval-topk=1.ckpt'):
        continue
    # Use the lineareval SimCLR wrapper to model -- use standard data transforms
    m1 = arch.create_model(model, source_dataset, pretrained=pretrained,
                        checkpoint_path=checkpoint_path, seed=seed,
                        callback=partial(LinearEvalWrapper,
                                        max_epochs,
                                        mean=DATASET_PARAMS[source_dataset]['mean'],
                                        std=DATASET_PARAMS[source_dataset]['std'],
                                        dataset_name=target_dataset))
    name, module = list(m1.model.named_modules())[-1]
    m1.model.__setattr__(name,
                         nn.Linear(module.in_features, 
                                   DATASET_PARAMS[target_dataset]['num_classes']))

    pl_utils.seed.seed_everything(seed, workers=True)

    fname = checkpoint_path.split('.ckpt')[0]
    checkpointer = NicerModelCheckpointing(
                                save_partial=[f'model.{x[0]}' for x in list(m1.model.named_parameters())[-2:]],
                                dirpath=dirpath, 
                                filename=f'{fname}_{target_dataset}_linear_eval', 
                                save_top_k=1,
                                every_n_epochs=0,
                                save_last=False,
                                verbose=True,
                                mode='max', # change to max if accuracy is being monitored
                                monitor='val_acc1')
    trainer = Trainer(accelerator='gpu', devices=devices,
                    num_nodes=num_nodes,
                    log_every_n_steps=1,
                    auto_select_gpus=True, 
                    deterministic=True,
                    max_epochs=max_epochs,
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=0,
                    callbacks=[LitProgressBar(['loss', 'train_acc1']),
                                checkpointer])
    # ## always use ddp -- works much faster, does not split batches
    trainer.fit(m1, datamodule=dm)
    # acc = trainer.test(m1, datamodule=dm)
    # print (f'Accuracy: {acc[0]["test_acc1"]}')
