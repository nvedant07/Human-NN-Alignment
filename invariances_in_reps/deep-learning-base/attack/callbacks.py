from typing import Union, Optional, Any
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as ch
from torch.nn.modules.loss import _Loss
from attack.attack_steps import AttackerStep
from attack.losses import BaseLoss, CompositeLoss

class AdvCallback(Callback):
    """
    Callback passed to Trainer to (dynamically) set attack params.
    This allows a general Attacker module to be defined for each model
    which can be resused by setting different attack params using this
    callback.
    """
    def __init__(self,
                 constraint_train: Union[str, AttackerStep],
                 eps_train: float,
                 step_size: float,
                 iterations_train: int,
                 random_start_train: bool = False,
                 random_restarts_train: int = 0,
                 constraint_val: Union[str, AttackerStep] = None,
                 eps_val: Optional[float] = None,
                 iterations_val: Optional[int] = None,
                 random_start_val: bool = False,
                 random_restarts_val: int = 0,
                 constraint_test: Union[str, AttackerStep] = None,
                 eps_test: Optional[float] = None,
                 iterations_test: Optional[int] = None,
                 random_start_test: bool = False,
                 random_restarts_test: int = 0,
                 do_tqdm: bool = False,
                 targeted: bool = False,
                 custom_loss: Union[CompositeLoss, BaseLoss, _Loss] = None, 
                 should_normalize: bool = True,
                 orig_input: Optional[ch.Tensor] = None, 
                 use_best: bool = True, 
                 return_image: bool = True,
                 est_grad: Optional[tuple] = None, 
                 mixed_precision: bool = False, 
                 model2: ch.nn.Module = None, 
                 targ2: ch.Tensor = None):
        ## See attack_module.Attacker's forward function for details of args
        super().__init__()
        self.constraint_train = constraint_train
        self.constraint_val = constraint_val if constraint_val is not None \
            else constraint_train
        self.constraint_test = constraint_test if constraint_test is not None \
            else constraint_train
        self.eps_train = eps_train
        self.eps_val = eps_val if eps_val is not None \
            else eps_train
        self.eps_test = eps_test if eps_test is not None \
            else eps_train
        self.step_size = step_size
        self.iterations_train = iterations_train
        self.iterations_val = iterations_val if iterations_val is not None \
            else iterations_train
        self.iterations_test = iterations_test if iterations_test is not None \
            else iterations_train
        self.random_start_train = random_start_train
        self.random_start_val = random_start_val if random_start_val is not None \
            else random_start_train
        self.random_start_test = random_start_test if random_start_test is not None \
            else random_start_train
        self.random_restarts_train = random_restarts_train
        self.random_restarts_val = random_restarts_val if random_restarts_val is not None \
            else random_restarts_train
        self.random_restarts_test = random_restarts_test if random_restarts_test is not None \
            else random_restarts_train
        self.do_tqdm = do_tqdm
        self.targeted = targeted
        self.custom_loss = custom_loss
        self.should_normalize = should_normalize
        self.orig_input = orig_input
        self.use_best = use_best
        self.return_image = return_image
        self.est_grad = est_grad
        self.mixed_precision = mixed_precision
        self.model2 = model2
        self.targ2 = targ2
    
    def get_eval_kwargs(self):
        return {
            'constraint': self.constraint_test,
            'eps': self.eps_test, 
            'step_size': self.step_size, 
            'iterations': self.iterations_test,
            'random_start': self.random_start_test, 
            'random_restarts': self.random_restarts_test, 
            'do_tqdm': self.do_tqdm,
            'targeted': self.targeted, 
            'custom_loss': self.custom_loss, 
            'should_normalize': self.should_normalize,
            'orig_input': self.orig_input, 
            'use_best': self.use_best, 
            'return_image': self.return_image, 
            'est_grad': self.est_grad, 
            'mixed_precision': self.mixed_precision, 
            'model2': self.model2, 
            'targ2': self.targ2
        }

    def get_train_kwargs(self):
        return {
            'constraint': self.constraint_train,
            'eps': self.eps_train, 
            'step_size': self.step_size, 
            'iterations': self.iterations_train,
            'random_start': self.random_start_train, 
            'random_restarts': self.random_restarts_train, 
            'do_tqdm': self.do_tqdm,
            'targeted': self.targeted, 
            'custom_loss': self.custom_loss, 
            'should_normalize': self.should_normalize,
            'orig_input': self.orig_input, 
            'use_best': self.use_best, 
            'return_image': self.return_image, 
            'est_grad': self.est_grad, 
            'mixed_precision': self.mixed_precision, 
            'model2': self.model2, 
            'targ2': self.targ2
        }

    def get_val_kwargs(self):
        return {
            'constraint': self.constraint_val,
            'eps': self.eps_val, 
            'step_size': self.step_size, 
            'iterations': self.iterations_val,
            'random_start': self.random_start_val, 
            'random_restarts': self.random_restarts_val, 
            'do_tqdm': self.do_tqdm,
            'targeted': self.targeted, 
            'custom_loss': self.custom_loss, 
            'should_normalize': self.should_normalize,
            'orig_input': self.orig_input, 
            'use_best': self.use_best, 
            'return_image': self.return_image, 
            'est_grad': self.est_grad, 
            'mixed_precision': self.mixed_precision, 
            'model2': self.model2, 
            'targ2': self.targ2
        }

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        attack_kwargs = self.get_train_kwargs()
        pl_module.__setattr__('attack_kwargs', attack_kwargs)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ch.set_grad_enabled(True) # need gradients for constructing adv samples
        attack_kwargs = self.get_val_kwargs()
        pl_module.__setattr__('attack_kwargs', attack_kwargs)        
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ch.set_grad_enabled(True)
        attack_kwargs = self.get_eval_kwargs()
        pl_module.__setattr__('attack_kwargs', attack_kwargs)
    
    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_test_epoch_start(trainer, pl_module)
    
    