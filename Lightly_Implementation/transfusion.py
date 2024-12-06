
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



class TransFusion_Head(nn.Module):
    def __init__(self, feature_dim: int = 128,
                        num_layers: int = 5,
                        num_heads: int = 8,
                        ff_ratio: int = 4):
        super(TransFusion_Head, self).__init__()

        self.head = []
        for _ in range(num_layers):
            self.head.append(TransFusionBlock(feature_dim=feature_dim,
                                                num_heads=num_heads,
                                                ff_dim=feature_dim*ff_ratio,
                                                activation='GELU'))

        self.model = nn.Sequential(*self.head)

    def forward(self, x):
        x = self.model(x)

        return x

class TransFusionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_dim, activation):
        super(TransFusionBlock, self).__init__()

        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)

        if activation == 'ReLU':
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, feature_dim)
            )
        elif activation == 'GELU':
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, feature_dim)
            )
        else:
            raise ValueError(f'Activation "{activation}" is not implemented')


        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x, attn_imaging = False):
        attn_output, self.attn_output_weights = self.attention(x, x, x, need_weights=attn_imaging, average_attn_weights=False)

        # Residual connection and first normalization
        x = x + attn_output
        x = self.norm1(x)

        # Second residual connection and normalization
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x
