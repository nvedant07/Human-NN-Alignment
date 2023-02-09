import itertools
import math
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.optim import SGD, lr_scheduler
import torchmetrics
from typing import Dict, Optional, Callable, Union
from datasets import dataset_metadata as ds
from attack.attack_module import Attacker
from architectures.utils import AverageMeter, InputNormalize
from architectures.inference import inference_with_features
from .lars import LARS


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class CompositeModule(nn.Module):
    """
    Combines model and projection in a single module
    can be passed to inference.inference_with_features
    Output is a tuple of (output, rep)
    output can be ignored but it's kept for compatibility
    """

    def __init__(self, model: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.projection = projection

    def forward(self, x: torch.Tensor, with_latent: bool=False, fake_relu: bool=False, 
                no_relu: bool=False, layer_num: Optional[int]=None):
        op, rep = inference_with_features(self.model, x, with_latent, 
                                         fake_relu, no_relu, layer_num)
        return op, self.projection(rep)


class SimCLRWrapper(LightningModule):
    """
    Wraps a pytorch model (from timm or otherwise) in a PyTorch-Lightning like
    model that can then be trained using a PyTorchLightning Trainer. 

    Input Normalization is performed here, before input is fed to the model

    The loss function used is InfoNCE as implemented here:
    https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L26
    """

    def __init__(self, 
                 model: nn.Module, 
                 max_epochs: int, 
                 batch_size: int,
                 num_samples: int,
                 mean: Optional[torch.Tensor] = None, 
                 std: Optional[torch.Tensor] = None, 
                 optim: str = 'lars',
                 temperature: float = 0.5, 
                 views: int = 2,
                 lr: Optional[float] = 4.0, 
                 warmup_epochs: int = 10,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 feat_dim: int = 128,
                 num_nodes: int = 1,
                 gpus: int = 1,
                 adv_aug: bool = False,
                 weight_decay: Optional[float] = 1.0e-06, 
                 step_lr: Optional[float] = None, 
                 step_lr_gamma: Optional[float] = None, 
                 momentum: Optional[float] = None, 
                 exclude_bn_bias: bool = False,
                 dataset_name: Optional[str] = None):
        super().__init__()

        self.save_hyperparameters(
            ignore=['model', 'mean', 'std', 'dataset_name', 'max_epochs'])

        self.model = model
        ## hidden_mlp must be the same size as 
        self.projection = Projection(input_dim, hidden_dim, feat_dim)
        self.loss_meter = AverageMeter('Loss')
        self.max_epochs = max_epochs

        if not (mean and std):
            mean = ds.DATASET_PARAMS[dataset_name]['mean']
            std = ds.DATASET_PARAMS[dataset_name]['std']
        self.normalizer = InputNormalize(mean, std)

        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size
        if adv_aug:
            self.attacker = Attacker(
                CompositeModule(self.model, self.projection), 
                self.normalizer)

    def forward(self, x, *args, **kwargs):
        backbone_ft = inference_with_features(self.model, self.normalizer(x), 
                                              with_latent=True, *args, **kwargs)[-1]
        return self.projection(backbone_ft) # ensure that projection features are normalized
    
    def nt_xent_loss(self, out_1, out_2, eps=1e-6):
        ### ref: https://github.com/PyTorchLightning/lightning-bolts/blob/142cfb46f5ae8090812d0a244f7df90da4b681d6/pl_bolts/models/self_supervised/simclr/simclr_module.py#L223
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2
        
        # out: [views * batch_size, dim]
        # out_dist: [views * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [views * batch_size, views * batch_size * world_size]
        # neg: [views * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.hparams.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.ones_like(neg).fill_(
            torch.e ** (1 / self.hparams.temperature))
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [views * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.hparams.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def shared_step(self, batch):
        (x_views), y = batch
        assert len(x_views) == self.hparams.views

        if hasattr(self, 'attacker'):
            noise = torch.ones_like(x_views[0])
            torch.normal(mean=0., std=0.00001, size=noise.shape, out=noise)
            x_views = [self.attacker(x + noise, self(x).detach(), 
                                     **self.attack_kwargs)[0].detach() \
                                         for x in x_views]

        ops = [self(x) for x in x_views]
        all_combinations = list(itertools.combinations(ops, 2))
        losses = torch.empty(len(all_combinations), device=ops[0].device)
        for idx, (op1, op2) in enumerate(all_combinations):
            losses[idx] = self.nt_xent_loss(op1, op2)
        return losses.mean()

    def training_step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        loss = self.shared_step(batch)
        self.loss_meter.update(loss.item(), len(batch[-1]))
        return {'loss': loss}

    def training_epoch_end(self, training_outputs):
        self.log("train_loss", self.loss_meter.avg)
        self.loss_meter.reset()

    def exclude_from_wt_decay(self):
        params = []
        excluded_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            elif not self.hparams.exclude_bn_bias:
                params.append(param)
            else:
                if 'bn' in name:
                    excluded_params.append(param)
                    continue
                if self.hparams.optim == 'lars' and 'bias' in name:
                    excluded_params.append(param)
                    continue
                params.append(param)

        return [
            {
                "params": params, 
                "weight_decay": self.hparams.weight_decay,
                "layer_adaptation": True
            },
            {
                "params": excluded_params,
                "weight_decay": 0.0,
                "layer_adaptation": False
            },
        ]

    def configure_optimizers(self):
        params = self.exclude_from_wt_decay()

        if self.hparams.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                trust_coefficient=0.001,
            )
        elif self.hparams.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        else:
            raise ValueError(f'{self.hparams.optim} not supported')

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
