import os
import yaml

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import Supervised_Cross_Entropy_Resnet

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

gpus = [2]

# -----------------------------------------------------------------------------

label_aug = {
    #'astro'                        : ["Astro-aug",              "astro_aug"],
    #'astro0'                       : ["Astro-aug-v0",           "astro_aug_v0"],
    #'astro2'                       : ["Astro-aug-v2",           "astro_aug_v2"],
    #'astro3'                       : ["Astro-aug-v3",           "astro_aug_v3"],
    #'astro4'                       : ["Astro-aug-v4",           "astro_aug_v4"],
    #'astro5'                       : ["Astro-aug-v5",           "astro_aug_v5"],
    #'astro6'                       : ["Astro-aug-v6",           "astro_aug_v6"],
    #'astro7'                       : ["Astro-aug-v7",           "astro_aug_v7"],
    #'astro8'                       : ["Astro-aug-v8",           "astro_aug_v8"],
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
    'rotation_v3'                  : ["Rotation-v3",            "rotation_v3"],
    #'blur'                         : ["Blur",                   "blur"],
    #'perspective'                  : ["Random perspective",     "pers"],
    #'rot_perspective'              : ["Rot-Perspective",        "rot_pers"],
    #'rot_perspective_blur'         : ["Rot-Perspective-Blur",   "rot_pers_blur"],
    #'grid_distortion'              : ["Grid distortion",        "grid"],
    #'rot_grid'                     : ["Rot-Grid",               "rot_grid"],
    #'rot_grid_blur'                : ["Rot-Grid-Blur",          "rot_grid_blur"],
    #'elastic_transform'            : ["Elastic transformation", "elastic"],
    #'rot_elastic'                  : ["Rot-Elastic",            "rot_elastic"],
    #'rot_elastic_blur'             : ["Rot-Elastic-Blur",       "rot_elastic_blur"],
    #'elastic_grid'                 : ["Elastic-Grid",           "elastic_grid"],
    #'elastic_prespective'          : ["Elastic-Perspective",    "elastic_pers"],
    #'grid_perspective'             : ["Grid-Perspective",       "grid_pers"],
    #'rot_elastic_grid_perspective' : ["Rot-Elastic-Grid-Pers",  "rot_elastic_grid_pers"]
    'without_aug'                  : ["Without-aug",       "without_aug"]
    }

label_features = {
    True: ["With features", "with_features"],
    False: ["Without features", "without_features"]
    }

# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)

# -----------------------------------------------------------------------------

def training_ce(hparams, name_checkpoint, name_tb, data_path):

    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.001,
        patience=70,
        mode="max",
        check_finite=True,
        divergence_threshold=0.1
    )

    # Save checkpoint
    checkpoint_callback = ModelCheckpoint_V2(
        monitor="accuracy_val",
        dirpath=os.path.join(config.model_path, "CE_resnet50_red"),
        filename=f"checkpoint_{name_checkpoint}",
        save_top_k=1,
        mode="max"
    )

    # Define the logger object
    tb_logger = TensorBoardLogger(
        save_dir='tb_logs',
        name='ce_resnet50_red',
        version=name_tb
    )

    # Initialize model
    model = Supervised_Cross_Entropy_Resnet(**hparams, data_path=data_path, resnet_model='resnet50')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=220,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        #progress_bar_refresh_rate=False,
        weights_summary=None
    )


    # Training
    trainer.fit(model)

    # Path of best model
    path = checkpoint_callback.best_model_path

    # Loads weights
    model = Supervised_Cross_Entropy_Resnet.load_from_checkpoint(path)

    # Load dataset
    model.prepare_data()

    # Compute metrics
    acc_val, conf_mat_val = model.confusion_matrix(dataset='Validation')
    acc_test, conf_mat_test = model.confusion_matrix(dataset='Test')

    return (acc_val, conf_mat_val), (acc_test, conf_mat_test)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the hyperparamters of the model
#hparams_file = open("results/hparams_best_ce_resnet.yaml", 'r')
hparams_file = open("results/hparams_best_ce.yaml", 'r')
hparams = Box(yaml.load(hparams_file, Loader=yaml.FullLoader))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

paths = {1: "dataset/td_ztf_stamp_17_06_20_sup_1.pkl",
         10: "dataset/td_ztf_stamp_17_06_20_sup_10.pkl",
         100: "dataset/td_ztf_stamp_17_06_20_sup_100.pkl"
         }

# -----------------------------------------------------------------------------

# Train for different augmentations
for augmentation in tqdm(label_aug.keys(), desc='Augmentations', unit= "aug"):
    for with_features in tqdm(label_features.keys(), leave=False, desc='Features', unit="feat"):
        for p in tqdm(paths.keys(), desc='dataset\'s fraction', unit= "p"):

            # Save accuracies and confusion matrixes for different initial conditions
            acc_array_val = []
            conf_mat_array_val = []
            acc_array_test = []
            conf_mat_array_test = []

            # Train for different initial conditions
            for rep in tqdm(range(5), leave=False, desc='Repetion', unit="rep"):

                hparams.augmentation = augmentation
                hparams.with_features = with_features

                # Train and compute metrics
                name_checkpoint = f"{p}_{augmentation}_{label_features[with_features][0]}_{rep+1}"
                name_tb = f"{p}_{augmentation}_{label_features[with_features][0]}"
                (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(hparams, name_checkpoint, name_tb, paths[p])

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
            title = f"""Confusion matrix ResNet50 classifier (labels {p}%)
({label_features[with_features][0]}, {label_aug[augmentation][0]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
            file = f"figures/confusion_matrix_CE_Resnet50-frac_{p}-Validation-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
            plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

            # Plot confusion matrix (test)
            # ----------------------------
            title = f"""Confusion matrix ResNet50 classifier (labels {p}%)
({label_features[with_features][0]}, {label_aug[augmentation][0]})
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
            file = f"figures/confusion_matrix_CE_Resnet50-frac_{p}-Test-{label_features[with_features][1]}-{label_aug[augmentation][1]}.png"
            plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
