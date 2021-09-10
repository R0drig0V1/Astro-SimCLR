import os
import pickle
import sys
import torch
import torchmetrics
import torchvision
import warnings

import numpy as np
import pytorch_lightning as pl

from utils.args import Args
from utils.training import Supervised_Cross_Entropy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

# Last version of pytorch is unstable
#warnings.filterwarnings("ignore")

# Sets seed
pl.utilities.seed.seed_everything(seed=1, workers=False)

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

# Configurations
config = Args({'num_nodes': 1,
               'gpus': 1,
               'workers': 4,
               'model_path': "../weights"
               })

# Hyperparameters
args = Args({'batch_size': 64,
             'image_size': 21,
             'max_epochs': 130,
             'drop_rate': 0.5,
             'optimizer': "SGD",
             'lr': 1e-3,
             'beta_loss': 0.2
             })

# -----------------------------------------------------------------------------

# Save checkpoint
checkpoint_callback = ModelCheckpoint(monitor="Accuracy",
                                      dirpath=os.path.join(config.model_path),
                                      filename="Supervised_Cross_Entropy-{epoch:02d}-{Accuracy:.2f}",
                                      save_top_k=1,
                                      mode="max")

# -----------------------------------------------------------------------------

# Defining the logger object
logger = TensorBoardLogger('tb_logs', name='Supervised_Cross_Entropy')

# -----------------------------------------------------------------------------

# Inicialize classifier
sce = Supervised_Cross_Entropy(image_size=args.image_size,
                               batch_size=args.batch_size,
                               drop_rate=args.drop_rate,
                               beta_loss=args.beta_loss,
                               lr=args.lr,
                               optimizer=args.optimizer)

# Trainer
trainer = pl.Trainer(max_epochs=args.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback],
                     logger=logger)

# Training
trainer.fit(sce)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback.best_model_path
print("\nBest model path:", path)

# Loads weights
sce = Supervised_Cross_Entropy.load_from_checkpoint(path)

# Load dataset and computes confusion matrixes
sce.prepare_data()
sce.conf_mat_val()
sce.conf_mat_test()

# -----------------------------------------------------------------------------
