import os
import yaml

import itertools as it
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR, SimCLR_classifier, Fine_SimCLR
from utils.repeater_simclr_lib import hyperparameter_columns

from box import Box
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

gpus = [2]

# -----------------------------------------------------------------------------

# Dataframe with results
df = pd.read_csv('results/hyperparameter_tuning_simclr.csv', index_col=0)

# Summary of hyperparameter tuning
df_summary = pd.read_csv('results/summary_tuning_simclr.csv', index_col=0)

# Names of hyperparameter columns of dataframe
hyper_columns = hyperparameter_columns(df)

# Hyperparameters to group the trials
hyper = ["config/encoder_name", "config/method", "config/astro_augmentation", "config/with_features"]

# -----------------------------------------------------------------------------

# labels for image names
label_encoder = {
    'pstamps': 'stamps',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50'
}

label_method ={
    'supcon': ['sup-simclr','sup_simclr'],
    'simclr': ['simclr', 'simclr']
}

label_aug = {
    True: ["astro-aug", "astro_aug"],
    False: ["default-aug", "default_aug"]
}

label_features = {
    True: ["with features", "with_feat"],
    False: ["without features", "without_feat"]
}

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Hyperparameters
args = Box({"image_size": 21,
            "batch_size": 75,
            "drop_rate": 0.2,
            "beta_loss": 0.512975,             
            "lr": 0.001503,
            "optimizer": "AdamW",
            "with_features": True,
            "balanced_batch": False,
            "augmentation": False
            })

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def training_fine_simclr(simclr_model, with_features):

    # Saves checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_val",
        dirpath=os.path.join(config.model_path),
        filename="finetuning_simclr-accuracy_val{accuracy_val:.3f}",
        save_top_k=1,
        mode="max"
    )


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.0008,
        patience=80,
        mode="max",
        check_finite=True,
        divergence_threshold=0.1
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='finetuning_simclr'
    )

    # Inicialize classifier
    fine_simclr = Fine_SimCLR(
        simclr,
        lr=args.lr,
        batch_size=args.batch_size,
        with_features=with_features
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=900,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )


    # Training
    trainer.fit(fine_simclr)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    fine_simclr = Fine_SimCLR.load_from_checkpoint(path, simclr_model=simclr_model)

    # Load dataset
    fine_simclr.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = fine_simclr.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = fine_simclr.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Dataframe of the best hyperparameters for each combination in hyper
df_best_simclr = df_summary.loc[df_summary.groupby(hyper)['mean'].idxmax()].reset_index(drop=True)
df_best_simclr.to_csv('results/summary_simclr.csv', float_format='%.6f')

for index in range(len(df_best_simclr)):

    # Best combination of hyperparameters       
    best_config = list(df_best_simclr.loc[index])


    # Mask of the best hyperparamter combination
    where = 1
    for hyper_name, opt_hyper in zip(hyper_columns, best_config):
        where = where & (df[hyper_name] == opt_hyper)

    # Folders of the best hyperparameter combination
    folders = list(df['logdir'][where])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
    for with_features in [True, False]:

        # Save accuracies and confusion matrixes for different initial conditions
        acc_array_val = []
        conf_mat_array_val = []
        acc_array_test = []
        conf_mat_array_test = []

        # Train for different initial conditions
        for exp_folder in folders:

            # Checkpoint path
            checkpoint_path = os.path.join(exp_folder, "checkpoint.ckpt")

            # Load weights
            simclr = SimCLR.load_from_checkpoint(checkpoint_path)

            # Train and compute metrics
            (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_fine_simclr(simclr, with_features)

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

        encoder_name = df_best_simclr.loc[index]["config/encoder_name"]
        method = df_best_simclr.loc[index]["config/method"]
        astro_augmentation = df_best_simclr.loc[index]["config/astro_augmentation"]


        # Plot confusion matrix (validation)
        # ---------------------------------
        title = f"""Confusion matrix SimCLR (fine-tuning)
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
        file = f"Figures/confusion_matrix_fine_SimCLR-Validation-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)


        # Plot confusion matrix (test)
        # ----------------------------
        title = f"""Confusion matrix SimCLR (fine-tuning)
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
        file = f"Figures/confusion_matrix_fine_SimCLR-Test-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
