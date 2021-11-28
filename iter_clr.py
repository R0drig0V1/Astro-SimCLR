import os

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR

from box import Box
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

gpus = [3]

# -----------------------------------------------------------------------------

def training_simclr(args_simclr):

    # Saves checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_val",
        dirpath=os.path.join(config.model_path),
        filename="simclr-accuracy_val{accuracy_val:.3f}",
        save_top_k=1,
        mode="max"
    )


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.0005,
        patience=200,
        mode="max",
        check_finite=True,
        divergence_threshold=0.05
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='simclr'
    )


    # Inicialize classifier
    simclr = SimCLR(
        encoder_name=args_simclr.encoder_name,
        method=args_simclr.method,
        image_size=args_simclr.image_size,
        astro_augmentation=args_simclr.astro_augmentation,
        projection_dim=args_simclr.projection_dim,
        temperature=args_simclr.temperature,
        lr_encoder=args_simclr.lr_encoder,
        batch_size_encoder=args_simclr.batch_size_encoder,
        optimizer_encoder=args_simclr.optimizer_encoder,
        beta_loss=args_simclr.beta_loss,
        lr_classifier=args_simclr.lr_classifier,
        batch_size_classifier=args_simclr.batch_size_classifier,
        optimizer_classifier=args_simclr.optimizer_classifier,
        with_features=args_simclr.with_features)
 

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args_simclr.max_epochs,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )


    # Training
    trainer.fit(simclr)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    sce = SimCLR.load_from_checkpoint(path)

    # Load dataset
    sce.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = sce.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = sce.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------

# Hyperparameters
args = Box({
    'max_epochs': 10,
    'encoder_name': 'pstamps',
    'method': 'supcon',
    'image_size': 21,
    'astro_augmentation': True,
    'projection_dim': 64,
    'temperature': 0.128,
    'lr_encoder': 1,
    'batch_size_encoder': 500,
    'optimizer_encoder': 'LARS',
    'beta_loss': 0.2,
    'lr_classifier': 1e-3,
    'batch_size_classifier': 100,
    'optimizer_classifier': 'SGD',
    'with_features':True
})

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_simclr(args)

    # Save metrics
    acc_array_val.append(acc_val)
    conf_mat_array_val.append(conf_mat_val)
    acc_array_test.append(acc_test)
    conf_mat_array_test.append(conf_mat_test)


# Compute mean and standard deviation of accuracy and confusion matrix
acc_mean_val = np.mean(acc_array_val, axis=0)
acc_std_val = np.std(acc_array_val, axis=0)
conf_mat_mean_val = np.mean(conf_mat_array_val, axis=0)
conf_mat_std_val = np.std(conf_mat_array_val, axis=0)

acc_mean_test = np.mean(acc_array_test, axis=0)
acc_std_test = np.std(acc_array_test, axis=0)
conf_mat_mean_test = np.mean(conf_mat_array_test, axis=0)
conf_mat_std_test = np.std(conf_mat_array_test, axis=0)


# Plot confusion matrix (validation)
# ---------------------------------
title = f"""Confusion matrix SimCLR classifier\n
            (with features, sup, astro-aug, Stamps)\n
            Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_Simclr-Stamps-Validation-with_features-sup-astro_aug.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix SimCLR classifier\n
            (with features, sup, astro-aug, Stamps)\n
            Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_Simclr-Stamps-Test-with_features-sup-astro_aug.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
