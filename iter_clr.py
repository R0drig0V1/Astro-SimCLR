import os
import yaml

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR_encoder_classifier, SimCLR_encoder_classifier_v2, SimCLR_encoder_classifier_2_datasets

from box import Box
from tqdm import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

gpus = [0]

# -----------------------------------------------------------------------------

# labels of encoders
label_encoder = {
    'pstamps': 'Stamps',
    'resnet18': 'Resnet18',
    'resnet34': 'Resnet34',
    'resnet50': 'Resnet50'
}

label_aug = {
    'astro'                        : ["Astro-aug",              "astro_aug"],
    'astro0'                       : ["Astro-aug-v0",           "astro_aug_v0"],
    'astro2'                       : ["Astro-aug-v2",           "astro_aug_v2"],
    'astro3'                       : ["Astro-aug-v3",           "astro_aug_v3"]
    #'astro4'                       : ["Astro-aug-v4",           "astro_aug_v4"],
    #'astro5'                       : ["Astro-aug-v5",           "astro_aug_v5"],
    #'astro6'                       : ["Astro-aug-v6",           "astro_aug_v6"]
    #'astro7'                       : ["Astro-aug-v7",           "astro_aug_v7"],
    #'astro8'                       : ["Astro-aug-v8",           "astro_aug_v8"]
    #'astro9'                       : ["Astro-aug-v9",           "astro_aug_v9"],
    #'simclr'                       : ["Simclr-aug",             "simclr_aug"],
    #'simclr2'                      : ["Simclr-aug-v2",          "simclr_aug_v2"],
    #'simclr3'                      : ["Simclr-aug-v3",          "simclr_aug_v3"]
    #'jitter_simclr'                : ["Jitter-simclr",          "jitter_simclr"],
    #'jitter_astro'                 : ["Jitter-astro",           "jitter_astro"],
    #'jitter_astro_v2'              : ["Jitter-astro v2",        "jitter_astro_v2"],
    #'jitter_astro_v3'              : ["Jitter-astro v3",        "jitter_astro_v3"],
    #'crop_simclr'                  : ["Crop-simclr",            "crop_simclr"],
    #'crop_astro'                   : ["Crop-astro",             "crop_astro"],
    #'rotation'                     : ["Rotation",               "rotation"],
    #'rotation_v2'                  : ["Rotation-v2",            "rotation_v2"],
    #'rotation_v3'                  : ["Rotation-v3",            "rotation_v3"],
    #'blur'                         : ["Blur",                   "blur"],
    #'perspective'                  : ["Random perspective",     "pers"],
    #'rot_perspective'              : ["Rot-Perspective",        "rot_pers"],
    #'rot_perspective_blur'         : ["Rot-Perspective-Blur",   "rot_pers_blur"]
    #'grid_distortion'              : ["Grid distortion",        "grid"],
    #'rot_grid'                     : ["Rot-Grid",               "rot_grid"],
    #'rot_grid_blur'                : ["Rot-Grid-Blur",          "rot_grid_blur"],
    #'elastic_transform'            : ["Elastic transformation", "elastic"]
    #'rot_elastic'                  : ["Rot-Elastic",            "rot_elastic"],
    #'rot_elastic_blur'             : ["Rot-Elastic-Blur",       "rot_elastic_blur"],
    #'elastic_grid'                 : ["Elastic-Grid",           "elastic_grid"],
    #'elastic_prespective'          : ["Elastic-Perspective",    "elastic_pers"],
    #'grid_perspective'             : ["Grid-Perspective",       "grid_pers"],
    #'rot_elastic_grid_perspective' : ["Rot-Elastic-Grid-Pers",  "rot_elastic_grid_pers"]
    }

label_features = {
    #True: ["With features", "with_features"],
    False: ["Without features", "without_features"]
    }

# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)

# -----------------------------------------------------------------------------

def training_simclr(hparams, rep):

    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="loss_train_enc",
        min_delta=0.002,
        patience=100,
        mode="min",
        check_finite=True
    )

    # Save checkpoint
    checkpoint_callback = ModelCheckpoint_V2(
        monitor="loss_train_enc",
        dirpath=os.path.join(config.model_path, "SimCLR_loss_encoder"),
        filename=f"checkpoint_{hparams.augmentation}_{rep}",
        save_top_k=1,
        mode="min"
    )

    # Define the logger object
    tb_logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='simclr_loss_encoder',
        version=hparams.augmentation
    )


    model = SimCLR_encoder_classifier_v2(**hparams,
                                         data_path="dataset/td_ztf_stamp_17_06_20.pkl")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=850,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        logger=tb_logger,
        #progress_bar_refresh_rate=True,
        weights_summary=None
    )


    # Training
    trainer.fit(model)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    model = SimCLR_encoder_classifier_v2.load_from_checkpoint(path)

    # Load dataset
    model.prepare_data_fast()

    # Compute metrics
    acc_val, conf_mat_val = model.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = model.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the hyperparamters of the model
#hparams_file = open("results/hparams_best_simclr.yaml", 'r')
#hparams = Box(yaml.load(hparams_file, Loader=yaml.FullLoader))
hparams = Box({
    'augmentation': "simclr",
    'batch_size_classifier': 60,
    'batch_size_encoder': 550,
    'beta_loss': 0.06421167982111942,
    'drop_rate_classifier': 0.2,
    'encoder_name': "pstamps",
    'image_size': 27,
    'lr_classifier': 0.019560409593991059,
    'lr_encoder': 0.258059613629328,
    'method': "simclr",
    'optimizer_classifier': "AdamW",
    'optimizer_encoder': "LARS",
    'projection_dim': 300,
    'temperature': 0.17910250439416514,
    'with_features': False
    })

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Train for different augmentations
for augmentation in tqdm(label_aug.keys(), desc='Augmentations', unit= "aug"):
    for with_features in tqdm(label_features.keys(), leave=False, desc='Features', unit="feat"):

        # Save accuracies and confusion matrixes for different initial conditions
        acc_array_val = []
        conf_mat_array_val = []
        acc_array_test = []
        conf_mat_array_test = []

        # Train for different initial conditions
        for rep in tqdm(range(5), leave=False, desc='Repetion', unit="rep"):

            hparams.augmentation = augmentation

            # Train and compute metrics
            (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_simclr(hparams, rep+1)

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
        title = f"""Confusion matrix (frozen encoder)
({label_features[with_features][0]}, {label_aug[augmentation][0]}, {label_encoder[hparams.encoder_name]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
        file = f"figures/confusion_matrix_SimCLR-Validation-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

        # Plot confusion matrix (test)
        # ----------------------------
        title = f"""Confusion matrix (frozen encoder)
({label_features[with_features][0]}, {label_aug[augmentation][0]}, {label_encoder[hparams.encoder_name]})
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
        file = f"figures/confusion_matrix_SimCLR-Test-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
        plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
