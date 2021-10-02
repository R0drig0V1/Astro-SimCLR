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
from utils.training import Self_Supervised_SimCLR, Linear_SimCLR, Self_Supervised_SimCLR_a, Self_Supervised_SimCLR_b

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

"""
# Hyperparameters self-supervised
args_ss_clr = Args({'batch_size': 50,
                    'image_size': 21,
                    'max_epochs': 5,
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
                   'max_epochs': 5,
                   'optimizer': "SGD",
                   'lr': 1e-3,
                   'beta_loss': 0.2,
                   'n_features': args_ss_clr.n_features
                   })
"""
# -----------------------------------------------------------------------------
"""
# Saves checkpoint
checkpoint_callback_ss = ModelCheckpoint(monitor="loss_val",
                                         dirpath=os.path.join(config.model_path),
                                         filename="Sim_CLR",
                                         save_top_k=1,
                                         mode="min")


# Defining the logger object
logger_ss = TensorBoardLogger('tb_logs', name='Sim_CLR')


# Early stop criterion
early_stop_callback_ss = EarlyStopping(monitor="loss_val",
                                       mode="min",
                                       patience=50,
                                       check_finite=True)

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
                     callbacks=[checkpoint_callback_ss,early_stop_callback_ss],
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
                                        filename="Linear_CLR",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_l = TensorBoardLogger('tb_logs', name='Linear_CLR')


# Early stop criterion
early_stop_callback_l = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=50,
                                      divergence_threshold=0.4,
                                      check_finite=True)

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
                     callbacks=[checkpoint_callback_l,early_stop_callback_l],
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

"""
"""
from losses import *



input_1 = torch.rand(10,10, dtype=torch.double, requires_grad=True)
input_2 = torch.rand(10,10, dtype=torch.double, requires_grad=True)
target = torch.randint(0, 5, (10,))
save = True
res = torch.autograd.gradcheck(SupConLoss(0.5), (input_1, input_2, target), raise_exception=False)
print(res) 
"""

# -----------------------------------------------------------------------------


# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 660,
                      'image_size': 21,
                      'max_epochs': 10,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.1,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 10,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2
                       })


# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='p_stamps',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=True)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 550,
                      'image_size': 63,
                      'max_epochs': 1000,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.5,
                      'n_features': 64,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 250,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2,
                      'n_features': args_ss_clr_a.n_features
                       })


# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='resnet18',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=True)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 550,
                      'image_size': 63,
                      'max_epochs': 1000,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.5,
                      'n_features': 64,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 250,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2,
                      'n_features': args_ss_clr_a.n_features
                       })


# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='resnet50',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=True)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 900,
                      'image_size': 21,
                      'max_epochs': 1000,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.5,
                      'n_features': 64,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 250,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2,
                      'n_features': args_ss_clr_a.n_features
                       })



# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='p_stamps',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=False)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 550,
                      'image_size': 63,
                      'max_epochs': 1000,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.5,
                      'n_features': 64,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 250,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2,
                      'n_features': args_ss_clr_a.n_features
                       })



# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='resnet18',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=False)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Hyperparameters self-supervised
args_ss_clr_a = Args({'batch_size': 550,
                      'image_size': 63,
                      'max_epochs': 1000,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'temperature': 0.5,
                      'n_features': 64,
                      'projection_dim': 64
                       })

# Hyperparameters linear classifier
args_ss_clr_b = Args({'batch_size': 100,
                      'image_size': args_ss_clr_a.image_size,
                      'max_epochs': 250,
                      'optimizer': "SGD",
                      'lr': 1e-3,
                      'drop_rate': 0.5,
                      'beta_loss': 0.2,
                      'n_features': args_ss_clr_a.n_features
                       })


# Saves checkpoint
checkpoint_callback_a = ModelCheckpoint(monitor="loss_val",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_a",
                                        save_top_k=1,
                                        mode="min")


# Defining the logger object
logger_a = TensorBoardLogger('tb_logs', name='Sim_CLR_a')


# Early stop criterion
early_stop_callback_a = EarlyStopping(monitor="loss_val",
                                      mode="min",
                                      patience=70,
                                      check_finite=True)


# Inicializes classifier
ss_clr_a = Self_Supervised_SimCLR_a(encoder_name='resnet50',
                                    image_size=args_ss_clr_a.image_size,
                                    batch_size=args_ss_clr_a.batch_size,
                                    projection_dim=args_ss_clr_a.projection_dim,
                                    temperature=args_ss_clr_a.temperature,
                                    lr=args_ss_clr_a.lr,
                                    optimizer=args_ss_clr_a.optimizer,
                                    method='supcon')
 
# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_a.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_a, early_stop_callback_a],
                     logger=logger_a)


# Training
trainer.fit(ss_clr_a)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_a.best_model_path
print("\nBest Self_Supervised_SimCLR path:", path)

# Loads weights
ss_clr_a = Self_Supervised_SimCLR_a.load_from_checkpoint(path)

# -----------------------------------------------------------------------------

# Saves checkpoint
checkpoint_callback_b = ModelCheckpoint(monitor="Accuracy",
                                        dirpath=os.path.join(config.model_path),
                                        filename="Sim_CLR_b",
                                        save_top_k=1,
                                        mode="max")


# Defining the logger object
logger_b = TensorBoardLogger('tb_logs', name='Sim_CLR_b')


# Early stop criterion
early_stop_callback_b = EarlyStopping(monitor="Accuracy",
                                      mode="max",
                                      min_delta=0.002,
                                      patience=70,
                                      divergence_threshold=0.4,
                                      check_finite=True)


# Inicialize classifier
sim_clr_b = Self_Supervised_SimCLR_b(simclr_model=ss_clr_a,
                                     image_size=args_ss_clr_b.image_size,
                                     batch_size=args_ss_clr_b.batch_size,
                                     beta_loss=args_ss_clr_b.beta_loss,
                                     lr=args_ss_clr_b.lr,
                                     drop_rate=args_ss_clr_b.drop_rate,
                                     optimizer=args_ss_clr_b.optimizer,
                                     with_features=False)


# Trainer
trainer = pl.Trainer(max_epochs=args_ss_clr_b.max_epochs,
                     gpus=config.gpus,
                     benchmark=True,
                     stochastic_weight_avg=False,
                     callbacks=[checkpoint_callback_b, early_stop_callback_b],
                     logger=logger_b)


# Training
trainer.fit(sim_clr_b)

# -----------------------------------------------------------------------------

# Path of best model
path = checkpoint_callback_b.best_model_path
print("\nBest Linear_CLR path:", path)

# Loads weights
sim_clr_b = Self_Supervised_SimCLR_b.load_from_checkpoint(path, simclr_model=ss_clr_a)

# Load dataset and computes confusion matrixes
sim_clr_b.prepare_data()
sim_clr_b.conf_mat_val()
sim_clr_b.conf_mat_test()