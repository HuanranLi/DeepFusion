import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NTXentLoss
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from io import BytesIO
import wandb
from PIL import Image
import numpy as np
from DataModule import *
from TransFusion import *
import numpy as np
import random

from lightning.pytorch import Trainer, seed_everything

def main(args):

    seed_everything(123, workers=True) #

    # torch.set_float32_matmul_precision('medium')
    if args.dataset == 'MNIST':
        data_module = MNIST_DataModule(input_size = args.input_size,
                                    batch_size = args.batch_size)
    elif args.dataset == 'CIFAR10':
        data_module = CIFAR10DataModule(input_size = args.input_size,
                                    batch_size = args.batch_size)
    elif args.dataset == 'CIFAR100':
        data_module = CIFAR100DataModule(input_size = args.input_size,
                                    batch_size = args.batch_size)
    elif args.dataset == 'Tiny_ImageNet':
        data_module = TinyImageNetDataModule(input_size = args.input_size,
                                    batch_size = args.batch_size)
    else:
        raise ValueError(f'Dataset {args.dataset} is not implemented!')

    # assert False
    # Initialize model
    model = TransFusionModel(num_layers=args.num_layers,
                                feature_dim=args.feature_dim,
                                output_size=args.output_size,
                                residual_attention=args.residual_attention,
                                residual_ffn=args.residual_ffn,
                                ff_dim=args.ff_dim,
                                num_heads=args.num_heads,
                                norm_type=args.norm_type,
                                lr=args.lr,
                                loss_temperature=args.loss_temperature,
                                att_logging_count=args.att_logging_count,
                                args=args)

    # Logger and checkpoint
    wandb_logger = WandbLogger(project="DeepFusion", config = args)

    # Trainer
    trainer = pl.Trainer(max_epochs=args.epochs,
                            accelerator=args.device,
                            logger=wandb_logger,
                            deterministic=True)

    trainer.fit(model, datamodule=data_module)
    data_module.setup('test')
    test_results = eval_knn(data_module.test_dataloader(), model, model.device)

    print(test_results)
    wandb.log({"test/Unsupervised_Accuracy": test_results})
    return model, test_results

# Setting up the command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TransFusion model on MNIST with wandb integration.")

    # Training
    parser.add_argument('--dataset', type=str, default='Tiny_ImageNet',
                            help='Choose the dataset to use: MNIST, CIFAR10 (default: MNIST).')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 256)')
    parser.add_argument('--input_size', type=int, default=32, help='input_size for image training (default: 32)')
    parser.add_argument('--loss_temperature', type=float, default=0.02, help='loss_temperature for contrastive loss')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=8e-5, help='learning rate (default: 8e-5)')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW', 'SGD'], default='SGD',
                        help='Choose the optimizer to use: Adam, AdamW, or SGD (default: SGD).')
    parser.add_argument('--scheduler', type=str, choices=['none', 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau'],
                        default='CosineAnnealingLR', help='Choose the scheduler to use: none, CosineAnnealingLR, ExponentialLR, or ReduceLROnPlateau (default: none).')
    parser.add_argument('--momentum', type=float, default=0.9, help='Set the momentum for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Set the weight decay for regularization.')

    # Model
    # TransFormer
    parser.add_argument('--residual_attention', type=float, default=1, help='coeff of residual connection for Attention Layer (0 for no residual)')
    parser.add_argument('--residual_ffn', type=float, default=1, help='coeff of residual connection for FFN layer (0 for no residual)')
    parser.add_argument('--feature_dim', type=int, default=256, help='feature_dim (default: 256)')
    parser.add_argument('--ff_dim', type=int, default=1024, help='feature_dim (default: 1024)')
    parser.add_argument('--num_layers', type=int, default=10, help='number of attention layers (default: 10)')
    parser.add_argument('--num_heads', type=int, default=16, help='number of attention head (default: 8)')
    parser.add_argument('--norm_type', type=str, default='layer', help='norm function to use (default: layer)')
    parser.add_argument('--activation', type=str, default='GELU', help='activation function to use (default: ReLU)')
    # Rest
    parser.add_argument('--backbone', type=str, default='resnet50', help='which backbone to train the images. (default: CNN3)')
    parser.add_argument('--output_size', type=int, default=512, help='output_size (default: 512)')
    parser.add_argument('--FFN_benchmark', type=int, default=0, help='Benchmark mode. Default 0-false')

    # Not for Sweep hparams
    parser.add_argument('--att_logging_count', type=int, default=0, help='Number of attention maps logged to the server during test phase. Default: 0')
    parser.add_argument('--device', type=str, default='auto', help='device to use')

    args = parser.parse_args()
    main(args)
