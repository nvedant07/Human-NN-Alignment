from typing import Optional, List
from pytorch_lightning.callbacks import progress
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer


class LitProgressBar(progress.TQDMProgressBar):

    def __init__(self, custom_metrics: Optional[List[str]] = []):
        """
        custom_metrics: list of metrics to display other than loss
        """
        super(LitProgressBar, self).__init__()
        self.enable = True
        self.custom_metrics = custom_metrics

    def enable(self):
        self.enable = True

    def disable(self):
        self.enable = False
    
    def add_custom_metrics(self, trainer: Trainer):
        new_quantities = {}
        for met in self.custom_metrics:
            if met in trainer.callback_metrics:
                new_quantities[met] = trainer.callback_metrics[met].item()
        return new_quantities

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule):
        ## do stuff here, do not override the method in LightningModule
        standard_metrics = pl_module.get_progress_bar_dict()
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            pl_utils.rank_zero.rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics, **self.add_custom_metrics(trainer)}

