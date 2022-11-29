import os
import yaml

import itertools as it
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR_encoder_classifier_2_datasets, SimCLR_classifier, Fine_SimCLR
from utils.repeater_lib import hyperparameter_columns

from box import Box
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)

# -----------------------------------------------------------------------------

# Dataframe of the best hyperparameters for each combination in hyper
trials = ["checkpoint_astro8_1.ckpt"]#, "checkpoint_astro2_2.ckpt", "checkpoint_astro2_3.ckpt"]#, "checkpoint_astro2_4.ckpt", "checkpoint_astro2_5.ckpt"]
folders = [f"/home/rvidal/weights/SimCLR_300_stamps/{trial}" for trial in trials]

# -----------------------------------------------------------------------------
    
# Train for different initial conditions
for rep, exp_folder in enumerate(folders):

    # Checkpoint path
    checkpoint_path = os.path.join(exp_folder)

    # Load weights
    simclr = SimCLR_encoder_classifier_2_datasets.load_from_checkpoint(checkpoint_path)

    # Load dataset
    simclr.prepare_data_fast()

    encoder_name = simclr.encoder_name
    augmentation = simclr.augmentation


    # Plot visualization (validation)
    # ---------------------------------
    file = f'figures/tsne_Validation_{encoder_name}_{augmentation}-td_ztf_stamp_300-{rep+1}.png'
    simclr.plot_tSNE('Validation', file, feats_in_plot=100)

    # Plot visualization (test)
    # ----------------------------
    file = f'figures/tsne_Test_{encoder_name}_{augmentation}-td_ztf_stamp_300-{rep+1}.png'
    simclr.plot_tSNE('Test', file, feats_in_plot=100)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
