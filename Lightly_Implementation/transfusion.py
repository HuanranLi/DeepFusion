
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


def get_normalization_layer(norm_type, num_features):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layer':
        return nn.LayerNorm(num_features)
    elif norm_type == 'group':
        return nn.GroupNorm(32, num_features)  # Example: 32 groups
    else:
        return None

class TransFusion_Head(nn.Module):
    def __init__(self, input_dim: int = 2048,
                        hidden_dim: int = 128,
                        output_dim: int = 128,
                        num_layers: int = 5):
        super(TransFusion_Head, self).__init__()
        # Construct layers with optional normalization
        self.pre_projector = nn.Linear(input_dim, hidden_dim)

        self.head = []
        for _ in range(num_layers):
            self.head.append(TransFusionBlock(feature_dim=hidden_dim,
                                                num_heads=8,
                                                ff_dim=input_dim*4,
                                                norm_type='layer',
                                                activation='GELU'))

        self.post_projector = nn.Linear(hidden_dim, output_dim)

        self.model = nn.Sequential(self.pre_projector, *self.head, self.post_projector)

    def forward(self, x):
        x = self.model(x)

        return x

class TransFusionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_dim, norm_type, activation):
        super(TransFusionBlock, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.norm_type = norm_type

        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = get_normalization_layer(norm_type, feature_dim)

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


        self.norm2 = get_normalization_layer(norm_type, feature_dim)

    def forward(self, x, attn_imaging = False):
        # Query, key, value generation
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_output, self.attn_output_weights = self.attention(q, k, v, need_weights=attn_imaging, average_attn_weights=False)

        # Residual connection and first normalization
        x = x + attn_output

        if self.norm1:
            x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Second residual connection and normalization
        x = x + ffn_output

        if self.norm2:
            x = self.norm2(x)

        return x
