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
from utils.training import Self_Supervised_SimCLR, Linear_SimCLR
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

# Hyperparameters self-supervised
args_ss_clr = Args({'batch_size': 300,
                    'image_size': 21,
                    'max_epochs': 150,
                    'drop_rate': 0.5,
                    'optimizer': "SGD",
                    'lr': 1e-3,
                    'temperature': 0.5,
                    'n_features': 64,
                    'projection_dim': 64
                    })

# Hyperparameters linear classifier
args_l_clr = Args({'batch_size': 100,
                   'image_size': args_ss_clr.image_size,
                   'max_epochs': 100,
                   'optimizer': "SGD",
                   'lr': 1e-3,
                   'beta_loss': 0.2,
                   'n_features': args_ss_clr.n_features
                   })

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_ss = ModelCheckpoint(monitor="loss_val",
                                         dirpath=os.path.join(config.model_path),
                                         filename="Sim_CLR--{epoch:02d}-{loss_val:.2f}",
                                         save_top_k=1,
                                         mode="max")


# Defining the logger object
logger_ss = TensorBoardLogger('tb_logs', name='Sim_CLR')


# Inicializes classifier
ss_clr = Self_Supervised_SimCLR(image_size=args_ss_clr.image_size,
                                batch_size=args_ss_clr.batch_size,
                                drop_rate=args_ss_clr.drop_rate,
                                n_features=args_ss_clr.n_features,
                                projection_dim=args_ss_clr.projection_dim,
                                temperature=args_ss_clr.temperature,
                                lr=args_ss_clr.lr,
                                optimizer=args_ss_clr.optimizer)

# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_ss],
                     logger=logger_ss)


# Training
trainer.fit(ss_clr)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_ss.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr = Self_Supervised_SimCLR.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_l = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Linear_CLR-{epoch:02d}-{Accuracy:.2f}",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_l = TensorBoardLogger('tb_logs', name='Linear_CLR')


# Inicialize classifier
l_clr = Linear_SimCLR(simclr_model=ss_clr,
                      image_size=args_l_clr.image_size,
                      batch_size=args_l_clr.batch_size,
                      n_features=args_l_clr.n_features,
                      beta_loss=args_l_clr.beta_loss,
                      lr=args_l_clr.lr,
                      optimizer=args_l_clr.optimizer)


# Trainer
trainer = pl.Trainer(max_epochs=args_l_clr.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_l],
                     logger=logger_l)


# Training
trainer.fit(l_clr)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_l.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
l_clr = Linear_SimCLR.load_from_checkpoint(path, simclr_model=ss_clr)

# Load dataset and computes confusion matrixes
l_clr.prepare_data()
l_clr.conf_mat_val()
l_clr.conf_mat_test()

# -----------------------------------------------------------------------------
