from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import _METRIC
from weakref import proxy
from typing import Dict, List, Optional
from bisect import bisect
import torch
import shutil


class NicerModelCheckpointing(ModelCheckpoint):

    def __init__(self, save_partial = [], *args, **kwargs):
        """
        Changes from ModelCheckpoint:
         * every_n_epochs (done after training): this is a positive int 
            which specifies which epochs should be checkpointed. 
            every_n_epochs = 1 means model is checkpointed after every epoch. 
            Saved as :
            {self.filename}{self.CHECKPOINT_JOIN_CHAR}epoch={epoch_num}{self.FILE_EXTENSION}
         * save_top_k (done after validation): this is an int which saves 
            the top k performing checkpoints. If this is zero then the quantity in ``monitor`` 
            is ignored and no topk models are saved, if 1 then the best
            performing model as per the quantity specified in ``monitor``is saved. 
            for save_top_k = -1, all model checkpoints are saved (in order).
            For any non-zero value, ``monitor`` must be specified.
            Saved as:
            {self.filename}{self.CHECKPOINT_JOIN_CHAR}topk={k}{self.FILE_EXTENSION}
         *  save_on_train_epoch_end has no meaning here
        """
        super(NicerModelCheckpointing, self).__init__(*args, **kwargs)
        self.save_partial = save_partial
    
    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            trainer.fast_dev_run  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        trainer.fit_loop.global_step -= 1
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._every_n_epochs > 0
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            self.save_checkpoint(trainer, flag='every_n_epochs')
        trainer.fit_loop.global_step += 1
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint when training stops.
        This will only save a checkpoint if `save_last` is also enabled as the monitor metrics logged during
        training/validation steps or end of epochs are not guaranteed to be available at this stage.
        """
        if self._should_skip_saving_checkpoint(trainer) or not self.save_last:
            return
        if self.verbose:
            rank_zero_info("Saving latest checkpoint...")
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        monitor_candidates = self._monitor_candidates(trainer, trainer.current_epoch, trainer.global_step - 1)
        trainer.fit_loop.global_step -= 1
        self._save_last_checkpoint(trainer, monitor_candidates)
        trainer.fit_loop.global_step += 1

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save a checkpoint at the end of the validation stage, 
        check for monitor metrics here
        """
        if self._should_skip_saving_checkpoint(trainer):
            return
        self.save_checkpoint(trainer, flag='topk')
    
    def save_final_checkpoint(self, trainer: Trainer, filepath: str):
        if len(self.save_partial) > 0:
            sd = {k:v for k,v in \
                trainer.training_type_plugin.model.state_dict().items() if k in self.save_partial}
            torch.save({'state_dict': sd}, filepath)
        else:
            trainer.save_checkpoint(filepath, self.save_weights_only)

    def save_checkpoint(self, trainer: Trainer, flag: Optional[str] = 'topk') -> None:
        """Performs the main logic around saving a checkpoint.
        This method runs on all ranks. It is the responsibility of `trainer.save_checkpoint` to correctly handle the
        behaviour in distributed training, i.e., saving only on rank 0 for data parallel use cases.
        """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self._validate_monitor_key(trainer)

        # track epoch when ckpt was last checked
        self._last_global_step_saved = global_step

        # what can be monitored
        monitor_candidates = self._monitor_candidates(trainer, epoch=epoch, step=global_step)

        # callback supports multiple simultaneous modes
        # here we call each mode sequentially
        if flag == 'topk':
            # Mode 1: save the top k checkpoints
            self._save_top_k_checkpoint(trainer, monitor_candidates)
        elif flag == 'every_n_epochs':
            # Mode 1: save the usual epochs
            self._save_every_n_epochs_checkpoint(trainer, monitor_candidates)
        # Mode 2: save last checkpoints
        self._save_last_checkpoint(trainer, monitor_candidates)

        # notify loggers
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.after_save_checkpoint(proxy(self))

    def _update_filepath(self, filepath: str, append: str) -> str:
        """Appends a string (append) to a given filepath"""
        i = filepath.split(self.FILE_EXTENSION)[0]
        return i + self.CHECKPOINT_JOIN_CHAR + append + self.FILE_EXTENSION

    def _save_every_n_epochs_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, _METRIC]) -> None:
        if self._every_n_epochs == 0 or self._every_n_epochs is None:
            return

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer)
        if 'epoch' not in filepath:
            filepath = self._update_filepath(filepath, f"epoch={trainer.current_epoch:d}")
        
        self.save_final_checkpoint(trainer, filepath)
        
        if self.verbose:
            rank_zero_info(f"Saved checkpoint for epoch: {trainer.current_epoch:d}")

    def _save_top_k_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, _METRIC]) -> None:
        if self.monitor is None or self.save_top_k == 0:
            return

        current = monitor_candidates.get(self.monitor)
        epoch = monitor_candidates.get("epoch")
        step = monitor_candidates.get("step")

        if self.check_monitor_top_k(trainer, current):
            k = self._update_best_and_save(current, trainer, monitor_candidates, self.save_partial)
            if self.verbose:
                rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor} saved with topk={k}")
        elif self.verbose:
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor} was not in top {self.save_top_k}")

    def _update_best_and_save(self, current: torch.Tensor, 
            trainer: Trainer, monitor_candidates: Dict[str, _METRIC],
            save_partial: List[str]
        ) -> int:

        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        
        if len(self.best_k_models) == 0:
            inserted_pos = 1
            filepath = self._update_filepath(filepath, f"topk={inserted_pos}")
        else:
            # find position of current item
            mnames, vals = list(zip(*sorted(self.best_k_models.items(), key=lambda x: x[1],
                                    reverse=self.mode!="min")))
            inserted_pos = bisect(vals, current) + 1
            filepath = self._update_filepath(filepath, f"topk={inserted_pos}")
            new_mnames = []
            for idx, m in enumerate(mnames):
                if idx < inserted_pos - 1:
                    new_mnames.append(m)
                else:
                    fname, extension = m.split('topk=')
                    updated_pos = int(extension.split(self.FILE_EXTENSION)[0]) + 1
                    new_mnames.append(f"{fname}topk={updated_pos}{self.FILE_EXTENSION}")
                if m == self.kth_best_model_path:
                    self.kth_best_model_path = new_mnames[-1]

            # update checkpoint names
            for old, new in zip(mnames, new_mnames):
                shutil.move(old,new)
            # update dict
            self.best_k_models = dict(zip(new_mnames, vals))

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
                f" (inserted at {inserted_pos})"
            )
        
        self.save_final_checkpoint(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            trainer.training_type_plugin.remove_checkpoint(del_filepath)
        
        return inserted_pos
