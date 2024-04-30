import math
from collections.abc import Iterable
from math import cos, floor, log, pi

import skorch
from torch.optim.lr_scheduler import _LRScheduler

_LRScheduler


class CyclicCosineDecayLR(skorch.callbacks.Callback):
    def __init__(
        self,
        optimizer,
        init_interval,
        min_lr,
        len_param_groups,
        base_lrs,
        restart_multiplier=None,
        restart_interval=None,
        restart_lr=None,
        last_epoch=-1,
    ):
        """
        Initialize new CyclicCosineDecayLR object
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_interval: (int) - Initial decay cycle interval.
        :param min_lr: (float or iterable of floats) - Minimal learning rate.
        :param restart_multiplier: (float) - Multiplication coefficient for increasing cycle intervals,
            if this parameter is set, restart_interval must be None.
        :param restart_interval: (int) - Restart interval for fixed cycle intervals,
            if this parameter is set, restart_multiplier must be None.
        :param restart_lr: (float or iterable of floats) - Optional, the learning rate at cycle restarts,
            if not provided, initial learning rate will be used.
        :param last_epoch: (int) - Last epoch.
        """
        self.len_param_groups = len_param_groups
        if restart_interval is not None and restart_multiplier is not None:
            raise ValueError(
                "You can either set restart_interval or restart_multiplier but not both"
            )

        if isinstance(min_lr, Iterable) and len(min_lr) != self.len_param_groups:
            raise ValueError(
                "Expected len(min_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(len(min_lr), self.len_param_groups)
            )

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(
            self.len_param_groups
        ):
            raise ValueError(
                "Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(len(restart_lr), self.len_param_groups)
            )

        if init_interval <= 0:
            raise ValueError(
                "init_interval must be a positive number, got {} instead".format(
                    init_interval
                )
            )

        group_num = self.len_param_groups
        self._init_interval = init_interval
        self._min_lr = [min_lr] * group_num if isinstance(min_lr, float) else min_lr
        self._restart_lr = (
            [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        )
        self._restart_interval = restart_interval
        self._restart_multiplier = restart_multiplier
        self.last_epoch = last_epoch
        self.base_lrs = base_lrs
        super().__init__()

    def on_batch_end(self, net, training, **kwargs):
        if self.last_epoch < self._init_interval:
            return self._calc(self.last_epoch, self._init_interval, self.base_lrs)

        elif self._restart_interval is not None:
            cycle_epoch = (
                self.last_epoch - self._init_interval
            ) % self._restart_interval
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch, self._restart_interval, lrs)

        elif self._restart_multiplier is not None:
            n = self._get_n(self.last_epoch)
            sn_prev = self._partial_sum(n)
            cycle_epoch = self.last_epoch - sn_prev
            interval = self._init_interval * self._restart_multiplier ** n
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch, interval, lrs)
        else:
            return self._min_lr

    def _calc(self, t, T, lrs):
        return [
            min_lr + (lr - min_lr) * (1 + cos(pi * t / T)) / 2
            for lr, min_lr in zip(lrs, self._min_lr)
        ]

    def _get_n(self, epoch):
        a = self._init_interval
        r = self._restart_multiplier
        _t = 1 - (1 - r) * epoch / a
        return floor(log(_t, r))

    def _partial_sum(self, n):
        a = self._init_interval
        r = self._restart_multiplier
        return a * (1 - r ** n) / (1 - r)


class LearningRateDecayCallback(skorch.callbacks.Callback):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.lr_warmup_end = config.lr_warmup_end
        self.lr_warmup_start = config.lr_warmup_start
        self.learning_rate = config.learning_rate
        self.warmup_batch = config.warmup_epoch * config.batch_per_epoch
        self.final_batch = config.final_epoch * config.batch_per_epoch

        self.batch_idx = 0

    def on_batch_end(self, net, training, **kwargs):
        """

        :param trainer:
        :type trainer:
        :param pl_module:
        :type pl_module:
        :param batch:
        :type batch:
        :param batch_idx:
        :type batch_idx:
        :param dataloader_idx:
        :type dataloader_idx:
        """
        # to avoid updating after validation batch
        if training:

            if self.batch_idx < self.warmup_batch:
                # linear warmup, in paper: start from 0.1 to 1 over lr_warmup_end batches
                lr_mult = float(self.batch_idx) / float(max(1, self.warmup_batch))
                lr = self.lr_warmup_start + lr_mult * (
                    self.lr_warmup_end - self.lr_warmup_start
                )
            else:
                # Cosine learning rate decay
                progress = float(self.batch_idx - self.warmup_batch) / float(
                    max(1, self.final_batch - self.warmup_batch)
                )
                lr = max(
                    self.learning_rate
                    + 0.5
                    * (1.0 + math.cos(math.pi * progress))
                    * (self.lr_warmup_end - self.learning_rate),
                    self.learning_rate,
                )
            net.lr = lr
            # for param_group in net.optimizer.param_groups:
            #   param_group["lr"] = lr

            self.batch_idx += 1


class LRAnnealing(skorch.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]["valid_loss_best"]:
            net.lr /= 4.0
