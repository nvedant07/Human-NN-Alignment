import torch
from typing import Optional
from .callbacks import AdvAttackWrapper
from .inference import inference_with_features

class InvertedRepWrapper(AdvAttackWrapper):

    def __init__(self, model: torch.nn.Module, seed: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.seed = seed # shape: (channels, width, height)

    def forward(self, x, *args, **kwargs):
        target_rep = inference_with_features(self.model, self.normalizer(x), *args)[1].detach()
        # .to(device) is unavoidable here, this way is much faster than putting a tensor of 
        # [batch, channels, wdith, height] on GPU
        seeds = torch.ones((len(x), *self.seed.shape), device=self.device) * \
            self.seed.unsqueeze(0).to(self.device)
        if 'custom_loss' in kwargs:
            if hasattr(kwargs['custom_loss'], 'scaling_factor'):
                if kwargs['custom_loss'].scaling_factor and \
                    kwargs['custom_loss'].scaling_factor < 0:
                    ## make it close to seed
                    kwargs['custom_loss']._set_target_inps(seeds)
                else:
                    kwargs['custom_loss']._set_target_inps(x)
        ir, _ = self.attacker(seeds, target_rep, **kwargs)
        return x, ir

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y  = batch
        # self.attack_kwargs['custom_loss']._set_target_inps(x)
        og, inverted_rep = self(x, True, **self.attack_kwargs)
        og, inverted_rep = og.detach(), inverted_rep.detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_inverted_reps = self.all_gather(inverted_rep)
            all_inverted_reps = torch.cat([all_inverted_reps[i] for i in \
                range(len(all_inverted_reps))])
            all_og = self.all_gather(og)
            all_og = torch.cat([all_og[i] for i in range(len(all_og))])
            return all_og, all_inverted_reps
        self.attack_kwargs['custom_loss'].clear_cache()
        return og, inverted_rep, y

    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_og, cat_ir, cat_y = None, None, None
            for batch_res in results[i]:
                og, ir, y = batch_res
                cat_og = og if cat_og is None else torch.cat((cat_og, og))
                cat_ir = ir if cat_ir is None else torch.cat((cat_ir, ir))
                cat_y = y if cat_y is None else torch.cat((cat_y, y))
            results[i] = (cat_og, cat_ir, cat_y)
        return results

