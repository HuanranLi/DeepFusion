import os
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
from DataModule import *
from TransFusion import *
import numpy as np

# mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class SimpleMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, args):
        super(SimpleMLP, self).__init__()
        layers = []
        if args.num_layers == 3:
            layers.append(nn.Linear(input_dim, 256))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(256, 128))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(128, output_dim))
        elif args.num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def load_CL_model(CL_model):
        self.CL_model = CL_model

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('FTtrain/loss', loss)
        self.log('FTtrain/acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('FTval/loss', loss)
        self.log('FTval/acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('FTtest/loss', loss)
        self.log('FTtest/acc', acc)
        return loss

    def create_optimizer(self, optimizer_type):
        if optimizer_type == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.args.lr)
        elif optimizer_type == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.args.lr, momentum=self.hparams.args.momentum, weight_decay=self.hparams.args.weight_decay)
        else:
            raise ValueError("Unsupported optimizer type")

    def create_scheduler(self, optimizer, scheduler_type):
        if not scheduler_type:
            return None

        if scheduler_type == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.args.epochs)
        elif scheduler_type == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        else:
            return None

    def configure_optimizers(self):
        optimizer = self.create_optimizer(self.hparams.args.optimizer)
        scheduler = self.create_scheduler(optimizer, self.hparams.args.scheduler)

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "FTval/loss"}
        else:
            return optimizer
