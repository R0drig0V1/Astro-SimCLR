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
from utils.config import config
from utils.training import Self_Supervised_SimCLR, Linear_SimCLR, CLR_a, CLR_b

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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

# Hyperparameters self-supervised
args_clr_a = Args({
    'encoder_name': 'resnet18',
    'method': 'supcon',
    'batch_size': 500,
    'image_size': 21,
    'max_epochs': 500,
    'optimizer': "LARS",
    'lr': 1e-3,
    'temperature': 0.5,
    'projection_dim': 64
})

# Hyperparameters linear classifier
args_clr_b = Args({
    'batch_size': 100,
    'image_size': args_clr_a.image_size,
    'max_epochs': 100,
    'optimizer': "SGD",
    'lr': 1e-3,
    'drop_rate': 0.5,
    'beta_loss': 0.2,
    'with_features': True,
})


# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(
    monitor="loss_val",
    dirpath=os.path.join(config.model_path),
    filename="clr_a-loss_val{loss_val:.3f}",
    save_top_k=1,
    mode="min"
)


# Defining the logger object
logger_a = TensorBoardLogger(
    save_dir='tb_logs',
    name='clr_a'
)


# Early stop criterion
early_stop_callback_a = EarlyStopping(
    monitor="loss_val",
    mode="min",
    patience=70,
    check_finite=True
)


# Inicializes classifier
clr_a = CLR_a(
    encoder_name=args_clr_a.encoder_name,
    image_size=args_clr_a.image_size,
    batch_size=args_clr_a.batch_size,
    projection_dim=args_clr_a.projection_dim,
    temperature=args_clr_a.temperature,
    lr=args_clr_a.lr,
    optimizer=args_clr_a.optimizer,
    method=args_clr_a.method
)
 
# Trainer
trainer = pl.Trainer(
    max_epochs=args_clr_a.max_epochs,
    gpus=config.gpus,
    benchmark=True,
    stochastic_weight_avg=False,
    callbacks=[checkpoint_callback_a, early_stop_callback_a],
    logger=logger_a
)


# Training
trainer.fit(clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest CLR path:", path)

# Loads weights
clr_a = CLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(
    monitor="accuracy_val",
    dirpath=os.path.join(config.model_path),
    filename="clr_b-acc_val{accuracy_val:.3f}",
    save_top_k=1,
    mode="max"
)


# Defining the logger object
logger_b = TensorBoardLogger(
    save_dir='tb_logs',
    name='clr_b'
)


# Early stop criterion
early_stop_callback_b = EarlyStopping(
    monitor="accuracy_val",
    mode="max",
    min_delta=0.002,
    patience=30,
    divergence_threshold=0.4,
    check_finite=True
)


# Inicialize classifier
clr_b = CLR_b(
    clr_model=clr_a,
    image_size=args_clr_b.image_size,
    batch_size=args_clr_b.batch_size,
    beta_loss=args_clr_b.beta_loss,
    lr=args_clr_b.lr,
    drop_rate=args_clr_b.drop_rate,
    optimizer=args_clr_b.optimizer,
    with_features=args_clr_b.with_features,
)


# Trainer
trainer = pl.Trainer(
    max_epochs=args_clr_b.max_epochs,
    gpus=config.gpus,
    benchmark=True,
    stochastic_weight_avg=False,
    callbacks=[checkpoint_callback_b, early_stop_callback_b],
    logger=logger_b
)


# Training
trainer.fit(clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest classifier path:", path)

# Loads weights
clr_b = CLR_b.load_from_checkpoint(path, clr_model=clr_a)

# Load dataset and computes confusion matrixes
clr_b.prepare_data()
clr_b.conf_mat_val()
clr_b.conf_mat_test()
