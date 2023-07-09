import os
import yaml

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR #SimCLR_encoder_classifier_2_datasets

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

# labels of encoders
label_encoder = {
    'pstamps': 'Stamps',
    'resnet18': 'Resnet18',
    'resnet34': 'Resnet34',
    'resnet50': 'Resnet50',
    'resnet152': 'Resnet152'
}

label_aug = {
    #'astro'                        : ["Astro-aug",              "astro_aug"],
    #'astro0'                       : ["Astro-aug-v0",           "astro_aug_v0"],
    #'astro2'                       : ["Astro-aug-v2",           "astro_aug_v2"],
    #'astro3'                       : ["Astro-aug-v3",           "astro_aug_v3"],
    #'astro4'                       : ["Astro-aug-v4",           "astro_aug_v4"],
    #'astro5'                       : ["Astro-aug-v5",           "astro_aug_v5"],
    #'astro6'                       : ["Astro-aug-v6",           "astro_aug_v6"],
    #'astro7'                       : ["Astro-aug-v7",           "astro_aug_v7"],
    'astro8'                       : ["Astro-aug-v8",           "astro_aug_v8"],
    #'astro9'                       : ["Astro-aug-v9",           "astro_aug_v9"],
    #'simclr'                       : ["Simclr-aug",             "simclr_aug"],
    #'simclr2'                      : ["Simclr-aug-v2",          "simclr_aug_v2"],
    #'simclr3'                      : ["Simclr-aug-v3",          "simclr_aug_v3"],
    #'jitter_simclr'                : ["Jitter-simclr",          "jitter_simclr"],
    #'jitter_astro'                 : ["Jitter-astro",           "jitter_astro"],
    #'jitter_astro_v2'              : ["Jitter-astro v2",        "jitter_astro_v2"],
    #'jitter_astro_v3'              : ["Jitter-astro v3",        "jitter_astro_v3"],
    #'crop_simclr'                  : ["Crop-simclr",            "crop_simclr"],
    #'crop_astro'                   : ["Crop-astro",             "crop_astro"],
    #'rotation'                     : ["Rotation",               "rotation"],
    #'rotation_v2'                  : ["Rotation-v2",            "rotation_v2"],
    #'rotation_v3'                  : ["Rotation-v3",            "rotation_v3"],
    #'blur'                         : ["Blur",                   "blur"],
    #'perspective'                  : ["Random perspective",     "pers"],
    #'rot_perspective'              : ["Rot-Perspective",        "rot_pers"],
    #'rot_perspective_blur'         : ["Rot-Perspective-Blur",   "rot_pers_blur"]
    #'grid_distortion'              : ["Grid distortion",        "grid"],
    #'rot_grid'                     : ["Rot-Grid",               "rot_grid"],
    #'rot_grid_blur'                : ["Rot-Grid-Blur",          "rot_grid_blur"],
    #'elastic_transform'            : ["Elastic transformation", "elastic"]
    #'rot_elastic'                  : ["Rot-Elastic",            "rot_elastic"],
    #'rot_elastic_blur'             : ["Rot-Elastic-Blur",       "rot_elastic_blur"],
    #'elastic_grid'                 : ["Elastic-Grid",           "elastic_grid"],
    #'elastic_prespective'          : ["Elastic-Perspective",    "elastic_pers"],
    #'grid_perspective'             : ["Grid-Perspective",       "grid_pers"],
    #'rot_elastic_grid_perspective' : ["Rot-Elastic-Grid-Pers",  "rot_elastic_grid_pers"]
    }

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
        max_epochs=800,
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
for augmentation in tqdm(label_aug.keys(), desc='Augmentations', unit= "aug"):
    for rep in tqdm(range(2,3), leave=False, desc='Repetion', unit="rep"):

        hparams.augmentation = augmentation
        training_simclr(hparams, rep)

# -----------------------------------------------------------------------------
