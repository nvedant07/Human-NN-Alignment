# quick testing script
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torchvision.transforms import transforms
from training import NicerModelCheckpointing, LitProgressBar
import torch
import architectures as arch
from attack.callbacks import AdvCallback
from attack.losses import LPNormLossSingleModel
from architectures.callbacks import LightningWrapper, AdvAttackWrapper
from architectures.inference import inference_with_features
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from self_supervised.simclr_datamodule import simclr_dm
from self_supervised.simclr_callback import SimCLRWrapper
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--seed', type=int, default=420)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--optim', type=str, default='lars')
parser.add_argument('--adv_aug', type=bool, default=False)
parser.add_argument('--exclude_bn_bias', type=int, choices=[0,1], default=0)
parser.add_argument('--nocolor', type=int, choices=[0,1], default=0)
args = parser.parse_args()


dataset = args.dataset
# dataset = 'imagenet'

# model = 'vgg16'
model = args.model
# model = 'resnet50'

checkpoint_path = ''

pretrained = False
seed = args.seed
devices = 2
num_nodes = 1
batch_size = args.batch_size if args.batch_size else DATASET_PARAMS[dataset]['batch_size']
max_epochs = args.max_epochs if args.max_epochs else DATASET_PARAMS[dataset]['epochs']
strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None

imagenet_path = <path>
data_path = <path>

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.,
    batch_size=batch_size)
# convert datamodule into a SimCLR datamodule
simclr_dm(dm, nocolor=bool(args.nocolor))


hidden_dim = inference_with_features(
    arch.create_model(model, dataset).eval(), 
    torch.rand(
        (1, 3, 
        DATASET_PARAMS[dataset]['input_size'], 
        DATASET_PARAMS[dataset]['input_size'])), with_latent=True)[-1].shape[-1]

# send SimCLRWrapper to model
m1 = arch.create_model(model, dataset, pretrained=pretrained,
                       checkpoint_path=checkpoint_path, seed=seed,
                       callback=partial(SimCLRWrapper,
                                        max_epochs=max_epochs,
                                        batch_size=dm.batch_size,
                                        num_samples=len(dm.train_ds),
                                        dataset_name=dataset,
                                        momentum=DATASET_PARAMS[dataset]['momentum'],
                                        adv_aug=args.adv_aug,
                                        gpus=devices,
                                        input_dim=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        exclude_bn_bias=bool(args.exclude_bn_bias),
                                        optim=args.optim))

pl_utils.seed.seed_everything(seed, workers=True)

dirpath = f'./checkpoints/{dataset}/'\
          f'{model}/simclr_bs_{dm.batch_size}_adv_{args.adv_aug}/'\
          f'{args.optim}_excludebn_{bool(args.exclude_bn_bias)}'
if bool(args.nocolor):
    dirpath += f'_nocolor'
checkpointer = NicerModelCheckpointing(
    dirpath=dirpath, 
    filename='{epoch}_rand_seed' + f'_{seed}', 
    every_n_epochs=50, 
    save_top_k=5, 
    save_last=False,
    verbose=True,
    mode='min', # change to max if accuracy is being monitored
    monitor='train_loss')

adv_callback = AdvCallback(constraint_train='2',
                           eps_train=1.,
                           step_size=.5,
                           iterations_train=5,
                           iterations_val=10,
                           iterations_test=10,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True,
                           do_tqdm=True,
                           custom_loss=LPNormLossSingleModel(lpnorm_type=2))
trainer = Trainer(accelerator='gpu', devices=devices,
                  num_nodes=num_nodes,
                  strategy=strategy, 
                  log_every_n_steps=1,
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=max_epochs,
                  check_val_every_n_epoch=1,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['loss']),
                             adv_callback,
                             checkpointer])
# ## always use ddp -- works much faster, does not split batches
trainer.fit(m1, datamodule=dm)
# acc = trainer.test(m1, datamodule=dm)
# print (f'Accuracy: {acc[0]["test_acc1"]}')
