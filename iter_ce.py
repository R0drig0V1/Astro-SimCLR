import os
import yaml
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import Supervised_Cross_Entropy

from utils.repeater_simclr_lib import hyperparameter_columns

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

gpus = [3]

# -----------------------------------------------------------------------------

# Dataframe with results
df = pd.read_csv('results/hyperparameter_tuning_ce.csv', index_col=0)

# Summary of hyperparameter tuning
df_summary = pd.read_csv('results/summary_tuning_ce.csv', index_col=0)

# Names of hyperparameter columns of dataframe
hyper_columns = hyperparameter_columns(df)

# -----------------------------------------------------------------------------

label_aug = {
    'astro': ["Astro-aug", "astro_aug"],
    'simclr': ["Simclr-aug", "simclr_aug"],
    'jitter_simclr': ["Jitter-simclr", "jitter_simclr"],
    'jitter_astro': ["Jitter-astro", "jitter_astro"],
    'crop_simclr': ["Crop-simclr", "crop_simclr"],
    'crop_astro': ["Crop-astro", "crop_astro"],
    'rotation': ["Rotation", "rotation"],
    'blur' : ["Blur", "blur"],
    'without_aug' : ["Without-aug", "without_aug"]
    }

label_features = {
    True: ["with features", "with_features"],
    False: ["without features", "without_features"]
    }

# -----------------------------------------------------------------------------

def training_ce(args):

    # Save checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_val",
        dirpath=os.path.join(config.model_path),
        filename="ce-accuracy_val{accuracy_val:.3f}",
        save_top_k=1,
        mode="max"
    )


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.002,
        patience=30,
        mode="max",
        check_finite=True,
        divergence_threshold=0.1
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='CE'
    )


    # Inicialize classifier
    sce = Supervised_Cross_Entropy(
        image_size=args.image_size,
        batch_size=args.batch_size,
        drop_rate=args.drop_rate,
        beta_loss=args.beta_loss,
        lr=args.lr,
        optimizer=args.optimizer,
        with_features=args.with_features,
        balanced_batch=args.balanced_batch,
        augmentation=args.augmentation
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=180,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )


    # Training
    trainer.fit(sce)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    sce = Supervised_Cross_Entropy.load_from_checkpoint(path)

    # Load dataset
    sce.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = sce.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = sce.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Dataframe of the best hyperparameters for each combination in hyper
df_best_ce = df_summary.loc[df_summary['mean'].idxmax()].reset_index(drop=True)

# Mask of the best hyperparamter combination
where = 1
for hyper_name, opt_hyper in zip(hyper_columns, df_best_ce):
    where = where & (df[hyper_name] == opt_hyper)

# Folders of the best hyperparameter combination
folder = list(df['logdir'][where])[0]

# Load the hyperparamters of the model
hparams_path = os.path.join(folder, "hparams.yaml")
hparams_file = open(hparams_path, 'r')
hparams = yaml.load(hparams_file, Loader=yaml.FullLoader)

# Copy the best hyperparameters's set
shutil.move(hparams_path, "results/hparams_best_ce.yaml")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Train for different augmentations
for augmentation in label_aug.keys():
    for with_features in label_features.keys():

        # Save accuracies and confusion matrixes for different initial conditions
        acc_array_val = []
        conf_mat_array_val = []
        acc_array_test = []
        conf_mat_array_test = []

        # Train for different initial conditions
        for _ in range(5):

            hparams['augmentation'] = augmentation
            hparams['with_features'] = with_features

            # Load weights
            sce = Supervised_Cross_Entropy(**hparams)

            # Train and compute metrics
            (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(sce)

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
        title = f"""Confusion matrix Stamps classifier
(without features, without augmentations)
({label_features[with_features][0]}, {label_aug[augmentation][0]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
        file = f"figures/confusion_matrix_CE-Validation-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

        # Plot confusion matrix (test)
        # ----------------------------
        title = f"""Confusion matrix Stamps classifier
({label_features[with_features][0]}, {label_aug[augmentation][0]})
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
        file = f"figures/confusion_matrix_CE-Test-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
