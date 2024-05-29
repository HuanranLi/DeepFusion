
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

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from torchvision.datasets import ImageFolder

from lightly.transforms.utils import IMAGENET_NORMALIZE
from tinyimagenet import TinyImageNet

class DataModule_Template(pl.LightningDataModule):
    def __init__(self, input_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size

        self.transform = None
        self.collate_fn = None

    def prepare_data(self):
        return

    def setup(self, stage=None):
        return


    def vanilla_training_loader(self):
        return DataLoader(self.vanilla_train_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 4,
                            persistent_workers=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=self.collate_fn,
                            num_workers = 4,
                            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 4,
                            persistent_workers=True)



class MNIST_DataModule(DataModule_Template):
    def __init__(self, input_size, batch_size):
        super().__init__(input_size, batch_size)
        self.transform = transforms.Grayscale(num_output_channels=3)
        self.collate_fn = SimCLRCollateFunction(input_size=self.input_size, gaussian_blur=0.,)

    def prepare_data(self):
        datasets.MNIST(root='../datasets', train=True, download=True, transform=self.transform)
        datasets.MNIST(root='../datasets', train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(root='../datasets', train=True, transform=self.transform)
            mnist_full_lightly = LightlyDataset.from_torch_dataset(mnist_full)
            self.train_dataset, self.val_dataset = random_split(mnist_full_lightly, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(root='../datasets', train=False, transform=transforms.Compose([self.transform,
                                                                                                            transforms.ToTensor()]))

            self.vanilla_train_dataset = datasets.MNIST(root='../datasets',
                                                    train=True,
                                                    transform=transforms.Compose([self.transform,
                                                                            transforms.ToTensor()]))

    def get_num_classes(self):
        return 10


class TinyImageNetDataModule(DataModule_Template):
    def __init__(self, input_size, batch_size):
        super().__init__(input_size, batch_size)

        self.transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]) ])

        self.collate_fn = SimCLRCollateFunction(input_size=self.input_size)

    def prepare_data(self):
        train_dataset = TinyImageNet(root = '../datasets/', split = 'train', download = True)
        test_dataset = TinyImageNet(root = '../datasets/', split = 'val', download = True)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = TinyImageNet(root = '../datasets/', split = 'train', download = False)
            train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
            self.train_size = len(train_dataset) - len(train_dataset) // 10
            self.val_size = len(train_dataset) // 10

            self.train_dataset, self.val_dataset = random_split(train_dataset, [self.train_size, self.val_size])
            print(self.train_size, self.val_size)

        if stage == 'test' or stage is None:
            self.test_dataset = TinyImageNet(root = '../datasets/', split = 'val', download = False, transform = self.transform)
            self.vanilla_train_dataset =TinyImageNet(root = '../datasets/', split = 'train', download = False, transform = self.transform)


    def get_num_classes(self):
        return 200



class CIFAR100DataModule(DataModule_Template):
    def __init__(self, input_size, batch_size):
        super().__init__(input_size, batch_size)

        self.transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]) ])


    def prepare_data(self):
        # Download CIFAR-10 dataset
        datasets.CIFAR100(root='../datasets', train=True, download=True)
        datasets.CIFAR100(root='../datasets', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load training dataset with transforms
            cifar_full = datasets.CIFAR100(root='../datasets', train=True)
            cifar_full_lightly = LightlyDataset.from_torch_dataset(cifar_full)
            self.train_size, self.val_size = len(cifar_full_lightly) - len(cifar_full_lightly) // 10, len(cifar_full_lightly) // 10
            print(self.train_size, self.val_size)

            self.train_dataset, self.val_dataset = random_split(cifar_full_lightly, [self.train_size, self.val_size])
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(root='../datasets', train=False, transform = self.transform)

            self.vanilla_train_dataset = datasets.CIFAR100(root='../datasets',
                                                         train=True, transform = self.transform)

    def get_num_classes(self):
        return 100


class CIFAR10DataModule(DataModule_Template):
    def __init__(self, input_size, batch_size):
        super().__init__(input_size, batch_size)
        self.transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]) ])

    def prepare_data(self):
        # Download CIFAR-10 dataset
        datasets.CIFAR10(root='../datasets', train=True, download=True)
        datasets.CIFAR10(root='../datasets', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load training dataset with transforms
            cifar_full = datasets.CIFAR10(root='../datasets', train=True)
            cifar_full_lightly = LightlyDataset.from_torch_dataset(cifar_full)

            self.train_size, self.val_size = len(cifar_full_lightly) - len(cifar_full_lightly) // 10, len(cifar_full_lightly) // 10
            # print(self.train_size, self.val_size)
            self.train_dataset, self.val_dataset = random_split(cifar_full_lightly, [self.train_size, self.val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(root='../datasets', train=False, transform = self.transform)
            self.vanilla_train_dataset = datasets.CIFAR10(root='../datasets',
                                                         train=True, transform = self.transform)


    def get_num_classes(self):
        return 10
