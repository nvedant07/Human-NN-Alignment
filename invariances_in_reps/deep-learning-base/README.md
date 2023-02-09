This is a directory for common DL code including models, adversarial attacks, supervised training. This is provides easy ways to extend models and training paradigms. This provides a smooth interface between many SOTA libraries like ``timm`` and ``robustness``. All the code relies on PyTorch and PyTorch-Lightning.

## Models

ImageNet (and similarly large) models are taken from [timm](https://github.com/rwightman/pytorch-image-models). 
CIFAR10/100 model schemas differ a little from ImageNet models.

Example of initializing a (random) model (in case dataset isn't specified in ``datasets.dataset_metadata``, initializes an ImageNet model):

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'

m = arch.create_model(model, dataset)
# m has randomly initialized weights with default seed
```

Example of initializing a (random) model with a different seed:

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'
seed = 42

m = arch.create_model(model, dataset, seed=seed)
# m has randomly initialized weights with seed=42
```

Initializing with user-defined pre-trained weights:

```python
import architectures as arch

dataset = 'imagenet'
model = 'resnet18'
pretrained = True
checkpoint_path = './checkpoint.best.pt'

m = arch.create_model(model, dataset, pretrained=pretrained, checkpoint_path=checkpoint_path)
# m takes weights from specified path
```

Initializing with pre-trained weights in ``timm`` (ImageNet only):

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'
pretrained = True

m = arch.create_model(model, dataset, pretrained=pretrained)
# m has pretrained weights as defined in timm
```

To perform inference (using PyTorch Lightning):

```python

from datasets.data_modules import DATA_MODULES
from architectures.callbacks import LightningWrapper
from functools import partial

dataset = 'imagenet'
model = 'resnet50'
devices = 2

m = arch.create_model(model, dataset, pretrained=pretrained, 
                      callback=partial(LightningWrapper, dataset_name=dataset))
dm = DATA_MODULES[dataset](data_dir='./data')

trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True)
acc = trainer.test(m1, datamodule=dm)
```

If you need predictions, implement the logic in `on_predict_step` hook of `LightningModule`. An example is provided in `AdvAttackWrapper`. See 


To get intermediate layer representations:

```python
# show an example of obtaining feature maps of ResNet
```


## Datasets

Standard dataloaders from torchvision + support for custom datasets. Many dataset classes taken from [robustness](https://github.com/MadryLab/robustness) and [robust-models-transfer](https://github.com/Microsoft/robust-models-transfer)

Initializing datasets:

```python

from datasets.data_modules import DATA_MODULES

dataset = 'imagenet'

dm = DATA_MODULES[dataset](data_dir='./data')
# dm is an instance of pl.LightningDataModule

# to access training data
for x, y in dm.train_dataloader():
    ...

# to access validation data
for x, y in dm.val_dataloader():
    ...

# to access test data
for x, y in dm.test_dataloader():
    ...

```


Datasets with custom data augmentations:

```python
# show the use of data augmentation callback
```


Model dependent data augmentations:

```python
# example of model dependent data aug, eg: worst-of-K spatial augmentations
```


## Training

Supports different losses (eg: adversarial training), different optimizers (all included in ``timm``)

Supervised: Uses [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for easy, scalable training. 

Self-supervised: Provides implementations under ``self_supervised``.

Uses [robustness](https://github.com/MadryLab/robustness) for attack module used in adversarial training.

Example of supervised training (standard):

```python
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from training import NicerModelCheckpointing, LitProgressBar
import architectures as arch
from architectures.callbacks import LightningWrapper
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from functools import partial

model = 'resnet50'
seed = 420
devices = 1

m1 = arch.create_model(model, dataset, pretrained=False, 
                       checkpoint_path='', seed=seed, 
                       callback=partial(LightningWrapper, dataset_name='cifar10'))
dm = DATA_MODULES['cifar10'](data_dir='./data')

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(dirpath=f'checkpoints/{dataset}/{model}', 
                               filename='{epoch}', 
                               every_n_epochs=5, 
                               save_top_k=5, 
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
                  max_epochs=DATASET_PARAMS[dataset]['epochs'],
                  check_val_every_n_epoch=1,
                  callbacks=[LitProgressBar(['running_acc']), 
                             checkpointer])
trainer.fit(m1, datamodule=dm)

```


Example of supervised training (adversarial):

```python
## bolierplate imports as above
from architectures.callbacks import AdvAttackWrapper

## define device, model name, dataset name etc.

m1 = arch.create_model(model, dataset, pretrained=pretrained, 
                       checkpoint_path=checkpoint_path, seed=seed, 
                       callback=partial(AdvAttackWrapper, dataset_name=dataset))

## setup datamodule as above, seed everything etc.
constraint, eps = '2', 1.
checkpointer = NicerModelCheckpointing(dirpath=f'checkpoints/{dataset}/{model}/AT_l{constraint}_eps{eps:.2f}', 
                               filename='{epoch}', 
                               every_n_epochs=5, 
                               save_top_k=5, 
                               save_last=False,
                               verbose=True,
                               mode='min', 
                               monitor='val_loss_adv')
adv_callback = AdvCallback(constraint_train=constraint,
                           eps_train=eps,
                           step_size=1.,
                           iterations_train=10,
                           iterations_val=100,
                           iterations_test=100,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True)
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=DATASET_PARAMS[dataset]['epochs'],
                  check_val_every_n_epoch=5,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['running_acc_clean', 'running_acc_adv']),
                             checkpointer,
                             adv_callback])
trainer.fit(m1, datamodule=dm)
```

Example of self-supervised learning (SimCLR) -- training of backbone:

```python
## bolierplate imports as above
from ssl.simclr_callback import SimCLRWrapper

dataset = 'cifar10'
model = 'vgg16'
checkpoint_path = ''
pretrained = False
seed = args.seed
devices = 8
num_nodes = 1
batch_size = 1024 ## simclr benefits from large batch sizes
max_epochs = 1000 ## simclr benefits from longer training
# set find_unused_parameters = True since we 
# ignore classification head in forward pass of SimCLRWrapper
strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None

imagenet_path = <path>
data_path = <path>

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.,
    batch_size=batch_size)
# convert datamodule into a SimCLR datamodule -- this generates multiple views
simclr_dm(dm, s=1, views=2)

# hidden dim is needed for projection head
hidden_dim = inference_with_features(
    arch.create_model(model, dataset).eval(), 
    torch.rand(
        (1, 3, 
        DATASET_PARAMS[dataset]['input_size'], 
        DATASET_PARAMS[dataset]['input_size'])), with_latent=True).shape[-1]

# send SimCLRWrapper to model
m1 = arch.create_model(model, dataset, pretrained=pretrained,
                       checkpoint_path=checkpoint_path, seed=seed,
                       callback=partial(SimCLRWrapper,
                                        max_epochs=max_epochs,
                                        batch_size=dm.batch_size,
                                        num_samples=len(dm.train_ds),
                                        dataset_name=dataset,
                                        optim='lars',
                                        lr=DATASET_PARAMS[dataset]['lr'],
                                        momentum=DATASET_PARAMS[dataset]['momentum'],
                                        weight_decay=DATASET_PARAMS[dataset]['weight_decay'],
                                        gpus=devices,
                                        input_dim=hidden_dim,
                                        hidden_dim=hidden_dim))

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(
                               dirpath=f'checkpoints/{dataset}/{model}/simclr_bs_{dm.batch_size}', 
                               filename='{epoch}_rand_seed' + f'_{seed}', 
                               every_n_epochs=50, 
                               save_last=False,
                               verbose=True)
trainer = Trainer(accelerator='gpu', devices=devices,
                  num_nodes=num_nodes,
                  strategy=strategy, 
                  log_every_n_steps=1,
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=max_epochs,
                  check_val_every_n_epoch=1,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['loss']),
                             checkpointer])
# ## always use ddp -- works much faster, does not split batches
trainer.fit(m1, datamodule=dm)
```

Example of self-supervised learning (SimCLR) -- linear eval:

```python
## bolierplate imports as above
from architectures.callbacks import LinearEvalWrapper

dataset = 'cifar10'
model = 'resnet50'

pretrained = True
seed = 1
devices = 1 # no need for multi-device training
num_nodes = 1
batch_size = DATASET_PARAMS[dataset]['batch_size']
max_epochs = DATASET_PARAMS[dataset]['epochs']

imagenet_path = <path>
data_path = <path>

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.,
    batch_size=batch_size)

checkpoint_path = f'checkpoints/epoch=959_rand_seed_{seed}.ckpt'

m1 = arch.create_model(model, dataset, pretrained=pretrained,
                        checkpoint_path=checkpoint_path, seed=seed,
                        callback=partial(LinearEvalWrapper,
                                        DATASET_PARAMS[dataset]['epochs'],
                                        dataset_name=dataset))

pl_utils.seed.seed_everything(seed, workers=True)

fname = checkpoint_path.split('.ckpt')[0]
checkpointer = NicerModelCheckpointing(
                            save_partial=[f'model.{x[0]}' for x in list(m1.model.named_parameters())[-2:]],
                            dirpath='checkpoints', 
                            filename=f'{fname}_linear_eval', 
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
trainer.fit(m1, datamodule=dm)
```

Example of self-supervised learning (BYOL):

```python
# show an example of BYOL
```


## Adversarial Attacks

Attack module in [robustness](https://github.com/MadryLab/robustness). Also includes spatial attacks, taken from [adversarial_spatial](https://github.com/MadryLab/adversarial_spatial).

To do adversarial attack on models, use the ``AdvAttackWrapper`` and simply perform inference on it.

```python
## boilerplate code as above

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.,
    subset=100,
    batch_size=32)
# val fraction = 0. makes val_ds = train_ds -- 
# use it to do inference on the training set

m1 = arch.create_model(model, dataset, pretrained=pretrained,
                       checkpoint_path=checkpoint_path, seed=seed, 
                       callback=partial(AdvAttackWrapper, 
                                        return_adv_samples=True,
                                        dataset_name=dataset))

adv_callback = AdvCallback(constraint_train='2',
                           eps_train=1.,
                           step_size=1.,
                           iterations_train=1,
                           iterations_val=10,
                           iterations_test=10,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True)

trainer = Trainer(accelerator='gpu', devices=devices,
                  num_nodes=num_nodes,
                  strategy=strategy, 
                  log_every_n_steps=1,
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=1,
                  check_val_every_n_epoch=1,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['loss', 'running_acc_clean', 'running_acc_adv']), 
                             adv_callback])
## trainer allows distributed inference
## DDP will spawn multiple processes
out = trainer.predict(m1, dataloaders=[dm.val_dataloader()]) 
# val_dataloader has the entire training set

if trainer.is_global_zero:
    ## do things on the main process
    for dl_wise_results in out: # out has results for each dataloader
        x, (x_adv, pred_x_adv) = dl_wise_results
```


To do spatial attacks:

```python
# Show an example of worst-of-K and first order spatial attacks
```

## Unit Tests

To run unit tests:

```bash
cd tests
pytest -c conftest.py \
--data_path="<path>" \
--imagenet_path="<path>"
```

Replace the paths with the actual paths