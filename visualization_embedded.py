import os

import pytorch_lightning as pl

from utils.training import SimCLR
from pytorch_lightning.callbacks import ModelCheckpoint

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

simclr_augs = ["astro2", "astro8", "astro9", "simclr", "simclr2"]
encoder = "resnet18"
image_size = 27

# -----------------------------------------------------------------------------

for simclr_aug in simclr_augs:
    for rep in range(3):

        # Load weights
        simclr = SimCLR.load_from_checkpoint(f"../weights/SimCLR_{encoder}_{image_size}_{simclr_aug}/checkpoint_{rep}.ckpt")

        # Load dataset
        simclr.prepare_data_fast()

        # Plot visualization (test)
        # ----------------------------
        file = f'figures/Visualization_Test_{encoder}_{image_size}_{simclr_aug}_{rep}.png'
        simclr.visualization('Test', file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
