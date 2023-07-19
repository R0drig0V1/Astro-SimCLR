import os
import yaml

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR

from box import Box
from tqdm import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"

# -----------------------------------------------------------------------------

gpus = [0]

# -----------------------------------------------------------------------------

augmentations = [
    #'astro',
    #'astro0',
    #'astro2',
    #'astro3',
    #'astro4',
    #'astro5',
    #'astro6',
    'astro7',
    #'astro8',
    #'astro9',
    #'simclr',
    'simclr2',
    #'simclr3',
    ]

# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)

# -----------------------------------------------------------------------------

def training_simclr(hparams, rep):

    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="loss_train_enc",
        min_delta=0.002,
        patience=100,
        mode="min"
    )

    # Save checkpoint
    checkpoint_callback = ModelCheckpoint_V2(
        monitor="loss_train_enc",
        dirpath=os.path.join(config.model_path, f"SimCLR_{hparams.encoder_name}_{hparams.image_size}_{hparams.augmentation}"),
        filename=f"checkpoint_{rep}",
        save_top_k=1,
        mode="min"
    )

    # Define the logger object
    tb_logger = TensorBoardLogger(
        save_dir='tb_logs',
        name=f'Simclr',
        version=f"{hparams.encoder_name}_{hparams.image_size}_{hparams.augmentation}_{rep}"
    )

    # Initialize pytorch_lightning module
    model = SimCLR(
        **hparams,
        data_path_simclr='dataset/td_ztf_stamp_simclr_300.pkl',
        data_path_classifier='dataset/td_ztf_stamp_17_06_20_sup_1.pkl')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=400,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        logger=tb_logger,
        weights_summary=None
    )


    # Training
    trainer.fit(model)

    return None

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the hyperparamters of the model
hparams = Box({
    'augmentation': "simclr",
    'batch_size_encoder': 450,
    'encoder_name': "resnet18",
    'image_size': 27,
    'lr_encoder': 1.258059613629328,
    'method': "simclr",
    'optimizer_encoder': "LARS",
    'projection_dim': 300,
    'temperature': 0.17910250439416514,
    })

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Train for different augmentations
for augmentation in tqdm(augmentations, desc='Augmentations', unit= "aug"):
    for rep in tqdm(range(1,2), leave=False, desc='Repetion', unit="rep"):

        hparams.augmentation = augmentation
        training_simclr(hparams, rep)

# -----------------------------------------------------------------------------
