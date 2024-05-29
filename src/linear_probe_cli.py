import os
import wandb
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from DataModule import *
from TransFusion import *
from cli import main
import numpy as np
import random
from LinearProbing import *
from lightning.pytorch import Trainer, seed_everything

def download_run(run_id):
    api = wandb.Api()
    project_path = "huanran-research/TransFusion/"
    run = api.run(f'{project_path}/{run_id}')

    # os.makedirs(output_dir, exist_ok=True)
    # run.download(root=output_dir, replace=True)

    # Extract the arguments from the run config
    args = argparse.Namespace(**run.config)
    original_test_accuracy = run.summary['test/MOCO_test_accuracy']
    return args, original_test_accuracy


def match_acc(original_accuracy, reproduced_accuracy, epsilon):
    return (original_accuracy - reproduced_accuracy) < epsilon


class LP_Datamodule(pl.LightningDataModule):
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproduce TransFusion model training with wandb integration.")
    parser.add_argument('--run_id', type=str, required=True, help='Wandb run ID to reproduce')
    parser.add_argument('--num_classes', type=int, default = 10, help='number of class')


    parser.add_argument('--num_layers', type=int, default=1, help='number of layers to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW', 'SGD'], default='SGD',
                        help='Choose the optimizer to use: Adam, AdamW, or SGD (default: SGD).')
    parser.add_argument('--scheduler', type=str, choices=['none', 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau'],
                        default='none', help='Choose the scheduler to use: none, CosineAnnealingLR, ExponentialLR, or ReduceLROnPlateau (default: none).')
    parser.add_argument('--momentum', type=float, default=0.9, help='Set the momentum for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Set the weight decay for regularization.')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--inference', type=int, default=0, help='inference mode. Default 0-false')


    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for accuracy comparison')
    args = parser.parse_args()

    run_args, original_test_accuracy = download_run(args.run_id)
    model, test_results = main(run_args)

    reproduced_test_accuracy = test_results[0]['test/MOCO_test_accuracy']
    if match_acc(original_test_accuracy, reproduced_test_accuracy, args.epsilon):
        print(f"Reproduced model's test accuracy {reproduced_test_accuracy:.4f} is within epsilon {args.epsilon} of original test accuracy {original_test_accuracy:.4f}.")

        seed_everything(123, workers=True)

        # Logger and checkpoint
        wandb_logger = WandbLogger(project="TransFusion", config = args)
        # Set up ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='validation/loss',  # or any other metric you have like 'val_loss'
            save_top_k=1,  # Saves only the best checkpoint
            mode='min',  # `min` for minimizing metric, `max` for maximizing metric
            auto_insert_metric_name=False  # Prevents redundant metric names in filename
        )

        # Trainer
        trainer = pl.Trainer(max_epochs=args.epochs,
                                accelerator='auto',
                                logger=wandb_logger,
                                callbacks=[checkpoint_callback],)
                                deterministic=True)


        train_loader, val_loader = model.prep_LinearProbe_data(new_batch_size = args.batch_size, inference = args.inference)
        datamodule = LP_Datamodule(train_loader, val_loader)
        if args.inference:
            input_dim = run_args.feature_dim
        else:
            input_dim = run_args.output_size
        mlp_model = SimpleMLP(input_dim=input_dim, output_dim=args.num_classes, args = args)

        trainer.fit(mlp_model, datamodule)
        # trainer.test(mlp_model, dataloaders = test_loader)
