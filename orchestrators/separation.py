#
# Created on Tue Sep 21 2021
# Authored by Daniel Leong (test-dan-run)
#

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LightningSeparation(pl.LightningModule):

    def __init__(self, model, optim_fn, objective_fn, distributed=False, sample_rate=16000, *args, **kwargs):

        super(LightningSeparation, self).__init__()

        self.model = model
        self.optim_fn = optim_fn
        self.loss_fn = objective_fn

        self.distributed = distributed
        self.sample_rate = sample_rate

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        
        _optim, _scheduler, interval = self.optim_fn

        optimizer = _optim(self.parameters())
        scheduler = _scheduler(optimizer)

        output = {
            'optimizer': optimizer,
            'monitor': 'valid_loss',
            'lr_scheduler': scheduler,
            'interval': interval
        }

        return output