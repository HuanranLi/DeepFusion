
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
from KNN import *


# Function to sample a percentage of the feature bank
def sample_feature_bank(feature_bank, feature_labels, percentage):
    num_samples = int(len(feature_bank) * percentage)
    indices = torch.randperm(len(feature_bank))[:num_samples]
    return feature_bank[indices], feature_labels[indices]


class TransFusionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, residual_attention, residual_ffn, ff_dim, norm_type, activation):
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

        self.register_buffer('residual_attention', torch.tensor(residual_attention))
        self.register_buffer('residual_ffn', torch.tensor(residual_ffn))

    def forward(self, x, attn_imaging = False):
        # Query, key, value generation
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_output, self.attn_output_weights = self.attention(q, k, v, need_weights=attn_imaging, average_attn_weights=False)

        # Residual connection and first normalization
        x = x + self.residual_attention * attn_output

        if self.norm1:
            x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Second residual connection and normalization
        x = x + self.residual_ffn * ffn_output

        if self.norm2:
            x = self.norm2(x)

        return x



def get_normalization_layer(norm_type, num_features):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layer':
        return nn.LayerNorm(num_features)
    elif norm_type == 'group':
        return nn.GroupNorm(32, num_features)  # Example: 32 groups
    else:
        return None


class TransFusionModel(pl.LightningModule):
    def __init__(self, num_layers,
                        feature_dim,
                        num_heads,
                        residual_attention,
                        residual_ffn,
                        ff_dim,
                        output_size,
                        norm_type,
                        lr,
                        loss_temperature,
                        att_logging_count,
                        args):

        super().__init__()
        self.save_hyperparameters()


        # Define the CNN architecture
        if args.backbone == 'resnet18':
            resnet18 = models.resnet18()
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            self.backbone_projector = nn.Linear(resnet18.fc.in_features, feature_dim)
        elif args.backbone == 'resnet50':
            resnet50 = models.resnet50()
            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            self.backbone_projector = nn.Linear(resnet50.fc.in_features, feature_dim)
        elif args.backbone == 'CNN3':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # First convolutional layer
                nn.ReLU(),                                             # Activation function
                nn.MaxPool2d(kernel_size=2, stride=2),                 # Pooling layer

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Second convolutional layer
                nn.ReLU(),                                              # Activation function
                nn.MaxPool2d(kernel_size=2, stride=2),                  # Pooling layer

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Third convolutional layer
                nn.ReLU(),                                               # Activation function
                nn.AdaptiveAvgPool2d((1, 1))                             # Adaptive pooling to 1x1
            )
            # Define the projection layer
            self.backbone_projector = nn.Linear(256, feature_dim)  # Assuming output of last CNN layer is 256 channels
        else:
            raise ValueError(f'backbone {args.backbone} not implemented')

        # Construct layers with optional normalization
        self.transfusion = nn.ModuleList()
        for _ in range(num_layers):
            if args.FFN_benchmark:
                print('Benchmark Mode! Only FFN for projection head.')
                self.transfusion.append(nn.Linear(feature_dim, feature_dim))
                self.transfusion.append(nn.BatchNorm1d(feature_dim)) #, track_running_stats = False
                self.transfusion.append(nn.ReLU())
            else:
                self.transfusion.append(TransFusionBlock(feature_dim=feature_dim,
                                                    num_heads=num_heads,
                                                    residual_attention=residual_attention,
                                                    residual_ffn=residual_ffn,
                                                    ff_dim=ff_dim,
                                                    norm_type=norm_type,
                                                    activation=args.activation))

        self.transfusion_projector = nn.Linear(feature_dim, output_size)
        self.criterion = NTXentLoss(temperature=loss_temperature)

        self.nan_limit = 5
        self.nan_count = 0

    def forward(self, x, inference = False, attn_imaging = False):

        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.backbone_projector(x)
        if inference:
            return x

        for layer in self.transfusion:
            if self.hparams.args.FFN_benchmark:
                x = layer(x)
            else:
                x = layer(x, attn_imaging = attn_imaging)  # Apply each layer in sequence

        x = self.transfusion_projector(x)
        return x

    # def on_train_epoch_start(self):
    #     # Retrieve current learning rate from optimizer
    #     lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     # Log the learning rate
    #     self.log('train/learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def training_step(self, batch, batch_idx):
        (x0, x1), labels, _ = batch

        batch_size = x0.size(0)
        input = torch.cat((x0, x1))
        outputs = self(input)
        y0, y1 = outputs.chunk(2)
        loss = self.criterion(y0, y1)
        self.log("train/loss", loss, batch_size=batch_size)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     (x0, x1), labels, _ = batch
    #
    #     batch_size = x0.size(0)
    #     input = torch.cat((x0, x1))
    #     outputs = self(input)
    #     y0, y1 = outputs.chunk(2)
    #     loss = self.criterion(y0, y1)
    #     print(loss)
    #     self.log("validation/loss", loss, batch_size=batch_size)
    #
    #     return loss
    #
    # def on_validation_epoch_end(self):
    #     # Access the current epoch loss
    #     current_loss = self.trainer.callback_metrics.get('validation/loss')
    #     # Check if the loss is NaN
    #     if torch.isnan(current_loss):
    #         self.nan_count += 1
    #         print(f"Warning: NaN loss detected. NaN count = {self.nan_count}")
    #     else:
    #         self.nan_count = 0  # reset counter if the loss is not NaN
    #
    #     # Stop training if NaN loss happens for too many consecutive epochs
    #     if self.nan_count >= self.nan_limit:
    #         print("Stopping training due to NaN loss detected for 5 consecutive epochs.")
    #         self.trainer.should_stop = True

    # def on_test_start(self):
    #     MOCO_test_accuracy = eval_knn(self.trainer.datamodule.test_dataloader(), self, self.device)
    #     self.log('test/MOCO_test_accuracy', MOCO_test_accuracy, logger=True)




    def log_attention_weights(self, labels):
        # Sort the attention weights based on the labels
        sorted_indices = labels.argsort().detach().cpu().numpy()

        for idx, layer in enumerate(self.transfusion):
            if isinstance(layer, TransFusionBlock) and layer.attn_output_weights is not None:
                attn_weights = layer.attn_output_weights.detach().cpu().numpy()

                for i in range(attn_weights.shape[0]):
                    attn_weights_i = attn_weights[i]
                    sorted_attn_weights = attn_weights_i[np.ix_(sorted_indices, sorted_indices)]

                    # Plot the sorted attention weights
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(sorted_attn_weights, cmap='viridis')
                    plt.title(f"Sorted Attention Weights Layer {idx}")
                    plt.xlabel('Sorted Query Index')
                    plt.ylabel('Sorted Key Index')

                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = Image.open(buf)

                    # Log the heatmap image to wandb using PyTorch Lightning's logging
                    self.logger.experiment.log({f"attention_layer_{idx:02}_head_{i:02}": [wandb.Image(image, caption=f"attention_layer_{idx:02}_head_{i:02}")]})
                    plt.close()


    def configure_optimizers(self):
        optimizer = self.create_optimizer(self.hparams.args.optimizer)
        scheduler = self.create_scheduler(optimizer, self.hparams.args.scheduler)

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss"}
        else:
            return optimizer

    def create_optimizer(self, optimizer_type):
        if optimizer_type == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif optimizer_type == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.args.momentum, weight_decay=self.hparams.args.weight_decay)
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

    def extract_features(self, loader, inference):
        features = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                feature = self(x, inference)
                features.append(feature)
                labels.append(y)
        return torch.cat(features), torch.cat(labels)

    def prep_LinearProbe_data(self, new_batch_size, inference):
        self.eval()
        # Sample 10% of the training data
        train_dataset = self.trainer.datamodule.vanilla_train_dataset
        num_samples = int(0.1 * len(train_dataset))
        _, train_subset = random_split(train_dataset, [len(train_dataset) - num_samples, num_samples])

        train_subset_loader = DataLoader(train_subset, batch_size= self.trainer.datamodule.batch_size, shuffle=True, num_workers = 4, persistent_workers=True)
        train_features, train_labels = self.extract_features(train_subset_loader, inference)
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)

        train_loader = DataLoader(train_dataset, batch_size= new_batch_size, shuffle=True, num_workers = 4, persistent_workers=True)


        test_loader = self.trainer.datamodule.test_dataloader()
        test_features, test_labels = self.extract_features(test_loader, inference)
        test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size= new_batch_size, shuffle=False, num_workers = 4, persistent_workers=True)

        return train_loader, test_loader
