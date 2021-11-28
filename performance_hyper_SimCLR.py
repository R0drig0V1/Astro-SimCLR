import os
import yaml

import itertools as it
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR, SimCLR_classifier
from utils.repeater import hyperparameter_columns

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

gpus = [3]

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
    'supcon': ['self-sup','self_sup'],
    'simclr': ['simclr', 'simclr']
}

label_aug = {
    True: ["astro-aug", "astro_aug"],
    False: ["default-aug", "default_aug"],
    'Jitter_default': ["Jitter-default", "Jitter_default"],
    'Jitter_astro': ["Jitter-astro", "Jitter_astro"],
    'Crop_default': ["Crop-default", "Crop_default"],
    'Crop_astro': ["Crop-astro", "Crop_astro"],
    'Rotation': ["Rotation", "Rotation"]
}

label_features = {
    True: ["with features", "with_feat"],
    False: ["without features", "without_feat"]
}

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def training_simclr_classifier(simclr_model, with_features):

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
        min_delta=0.001,
        patience=70,
        mode="max",
        check_finite=True,
        divergence_threshold=0.1
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='simclr_classifier'
    )


    # Inicialize classifier
    simclr_classifier = SimCLR_classifier(simclr_model, with_features)

 
    # Trainer
    trainer = pl.Trainer(
        max_epochs=350,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )


    # Training
    trainer.fit(simclr_classifier)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    simclr_classifier = SimCLR_classifier.load_from_checkpoint(path, simclr_model=simclr_model)

    # Load dataset
    simclr_classifier.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = simclr_classifier.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = simclr_classifier.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def training_simclr(simclr):

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
        min_delta=0.0008,
        patience=100,
        mode="max",
        check_finite=True,
        divergence_threshold=0.1
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='simclr'
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
    trainer.fit(simclr)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    simclr = SimCLR.load_from_checkpoint(path)

    # Load dataset
    simclr.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = simclr.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = simclr.confusion_matrix(dataset='Test')

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

        # Load dataset and computes confusion matrixes
        simclr.prepare_data()

        # Compute metrics
        acc_val, conf_mat_val = simclr.confusion_matrix(dataset='Validation')
        acc_test, conf_mat_test = simclr.confusion_matrix(dataset='Test')

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
    with_features = df_best_simclr.loc[index]["config/with_features"]

    # Plot confusion matrix (validation)
    # ---------------------------------
    title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
    file = f"Figures/confusion_matrix_SimCLR-Validation-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
    plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)


    # Plot confusion matrix (test)
    # ----------------------------
    title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Test:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
    file = f"Figures/confusion_matrix_SimCLR-Test-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
    plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

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
        (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_simclr_classifier(simclr, with_features=False)

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
    with_features = False

    # Plot confusion matrix (validation)
    # ---------------------------------
    title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
    file = f"Figures/confusion_matrix_SimCLR-Validation-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
    plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)


    # Plot confusion matrix (test)
    # ----------------------------
    title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[astro_augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Test:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
    file = f"Figures/confusion_matrix_SimCLR-Test-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[astro_augmentation][1]}-{label_encoder[encoder_name]}.png"
    plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

    # Augmentations
    augmentations = ['Jitter_default', 'Jitter_astro', 'Crop_default', 'Crop_astro', 'Rotation']

    # Train for different augmentations
    for augmentation in augmentations:

        # Save accuracies and confusion matrixes for different initial conditions
        acc_array_val = []
        conf_mat_array_val = []
        acc_array_test = []
        conf_mat_array_test = []

        # Train for different initial conditions
        for exp_folder in folders:

            # Load the hyperparamters of the model
            hparams_path = os.path.join(exp_folder, "hparams.yaml")
            hparams_file = open(hparams_path, 'r')
            hparams = yaml.load(hparams_file, Loader=yaml.FullLoader)
            hparams['astro_augmentation'] = augmentation
            hparams['with_features'] = False

            # Load weights
            simclr = SimCLR(**hparams)

            # Train and compute metrics
            (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_simclr(simclr)

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
        with_features = False

        # Plot confusion matrix (validation)
        # ---------------------------------
        title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
        file = f"Figures/confusion_matrix_SimCLR-Validation-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[augmentation][1]}-{label_encoder[encoder_name]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)


        # Plot confusion matrix (test)
        # ----------------------------
        title = f"""Confusion matrix SimCLR classifier
({label_features[with_features][0]}, {label_method[method][0]}, {label_aug[augmentation][0]}, {label_encoder[encoder_name]})
Accuracy Test:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
        file = f"Figures/confusion_matrix_SimCLR-Test-{label_features[with_features][1]}-{label_method[method][1]}-{label_aug[augmentation][1]}-{label_encoder[encoder_name]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
