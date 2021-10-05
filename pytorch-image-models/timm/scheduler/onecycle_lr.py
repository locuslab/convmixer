""" OneCycle Scheduler
"""
import logging
import math
import numpy as np
import torch
from .scheduler import Scheduler

_logger = logging.getLogger(__name__)

class OneCycleLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)
        assert warmup_t == 0, "this schedule has warmup built in"
        assert t_initial > 0
        self.t_initial = t_initial
        
    def get_frac_epoch_values(self, frac_epoch: int):
        sched = lambda t, lr_max: np.interp([t], [0, self.t_initial*2//5, self.t_initial*4//5, self.t_initial], 
                                      [0, lr_max, lr_max/20.0, 0])[0]
        return [sched(frac_epoch, v) for v in self.base_values]
            
    def get_epoch_values(self, epoch: int):
        return self.get_frac_epoch_values(epoch)
