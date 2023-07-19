import os
import yaml

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import SimCLR_classifier, SimCLR

from box import Box
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

gpus = [0]

# -----------------------------------------------------------------------------

# labels for image names
label_encoder = {
    'pstamps': 'stamps',
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50'
}


label_aug = {
    #'astro'                        : ["Astro-aug",              "astro_aug"],
    #'astro0'                       : ["Astro-aug-v0",           "astro_aug_v0"],
    #'astro2'                       : ["Astro-aug-v2",           "astro_aug_v2"]
    #'astro3'                       : ["Astro-aug-v3",           "astro_aug_v3"],
    #'astro4'                       : ["Astro-aug-v4",           "astro_aug_v4"]
    #'astro5'                       : ["Astro-aug-v5",           "astro_aug_v5"],
    #'astro6'                       : ["Astro-aug-v6",           "astro_aug_v6"]
    #'astro7'                       : ["Astro-aug-v7",           "astro_aug_v7"]
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
    'without_aug'                  : ["Without-aug",            "without_aug"]
    }

label_aug2 = {
    'astro'                        : ["Astro-aug",              "astro_aug"],
    'astro0'                       : ["Astro-aug-v0",           "astro_aug_v0"],
    'astro2'                       : ["Astro-aug-v2",           "astro_aug_v2"],
    'astro3'                       : ["Astro-aug-v3",           "astro_aug_v3"],
    'astro4'                       : ["Astro-aug-v4",           "astro_aug_v4"],
    'astro5'                       : ["Astro-aug-v5",           "astro_aug_v5"],
    'astro6'                       : ["Astro-aug-v6",           "astro_aug_v6"],
    'astro7'                       : ["Astro-aug-v7",           "astro_aug_v7"],
    'astro8'                       : ["Astro-aug-v8",           "astro_aug_v8"],
    'astro9'                       : ["Astro-aug-v9",           "astro_aug_v9"],
    'simclr'                       : ["Simclr-aug",             "simclr_aug"],
    'simclr2'                      : ["Simclr-aug-v2",          "simclr_aug_v2"],
    'simclr3'                      : ["Simclr-aug-v3",          "simclr_aug_v3"],
    'jitter_simclr'                : ["Jitter-simclr",          "jitter_simclr"],
    'jitter_astro'                 : ["Jitter-astro",           "jitter_astro"],
    'jitter_astro_v2'              : ["Jitter-astro v2",        "jitter_astro_v2"],
    'jitter_astro_v3'              : ["Jitter-astro v3",        "jitter_astro_v3"],
    'crop_simclr'                  : ["Crop-simclr",            "crop_simclr"],
    'crop_astro'                   : ["Crop-astro",             "crop_astro"],
    'rotation'                     : ["Rotation",               "rotation"],
    'rotation_v2'                  : ["Rotation-v2",            "rotation_v2"],
    'rotation_v3'                  : ["Rotation-v3",            "rotation_v3"],
    'blur'                         : ["Blur",                   "blur"],
    'perspective'                  : ["Random perspective",     "pers"],
    'rot_perspective'              : ["Rot-Perspective",        "rot_pers"],
    'rot_perspective_blur'         : ["Rot-Perspective-Blur",   "rot_pers_blur"],
    'grid_distortion'              : ["Grid distortion",        "grid"],
    'rot_grid'                     : ["Rot-Grid",               "rot_grid"],
    'rot_grid_blur'                : ["Rot-Grid-Blur",          "rot_grid_blur"],
    'elastic_transform'            : ["Elastic transformation", "elastic"],
    'rot_elastic'                  : ["Rot-Elastic",            "rot_elastic"],
    'rot_elastic_blur'             : ["Rot-Elastic-Blur",       "rot_elastic_blur"],
    'elastic_grid'                 : ["Elastic-Grid",           "elastic_grid"],
    'elastic_prespective'          : ["Elastic-Perspective",    "elastic_pers"],
    'grid_perspective'             : ["Grid-Perspective",       "grid_pers"],
    'rot_elastic_grid_perspective' : ["Rot-Elastic-Grid-Pers",  "rot_elastic_grid_pers"],
    'without_aug'                  : ["Without-aug",            "without_aug"]
}

label_features = {
    #True: ["with features", "with_features"],
    False: ["without features", "without_features"]
    }

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the hyperparamters of the model
hparams_file = open("results/hparams_best_ce.yaml", 'r')
hparams = Box(yaml.load(hparams_file, Loader=yaml.FullLoader))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def training_simclr_classifier(simclr_model, augmentation, rep, p, with_features, data_path):

    # Saves checkpoint
    checkpoint_callback = ModelCheckpoint_V2(
        monitor="accuracy_val",
        dirpath=os.path.join(config.model_path, f"LC"),
        filename=f"checkpoint_{p}_{simclr.encoder_name}_{simclr.image_size}_{simclr.augmentation}_{augmentation}_{label_features[with_features][1]}_{rep}",
        save_top_k=1,
        mode="max"
    )


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.001,
        patience=40,
        mode="max"
    )


    # Define the logger object
    logger = TensorBoardLogger(
        save_dir='tb_logs',
        name=f'LC',
        version=f"{p}_{simclr.encoder_name}_{simclr.image_size}_{simclr.augmentation}_{augmentation}_{label_features[with_features][1]}_{rep}"
    )

    # Inicialize classifier
    simclr_classifier = SimCLR_classifier(
        simclr_model,
        batch_size=hparams.batch_size,
        drop_rate=hparams.drop_rate,
        beta_loss=hparams.beta_loss,
        lr=hparams.lr,
        optimizer=hparams.optimizer,
        with_features=with_features,
        augmentation=augmentation,
        data_path=data_path
    )

 
    # Trainer
    trainer = pl.Trainer(
        max_epochs=220,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        weights_summary=None
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

# Augmentations simclr
simclr_augs = ["astro8"]#"astro2", "astro9", "simclr"]#, "simclr2"]"astro8"
encoder = "resnet18"
image_size = 27

# -----------------------------------------------------------------------------

paths = {1: "dataset/td_ztf_stamp_17_06_20_sup_1.pkl",
         10: "dataset/td_ztf_stamp_17_06_20_sup_10.pkl",
         100: "dataset/td_ztf_stamp_17_06_20.pkl"
         }

# -----------------------------------------------------------------------------
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
            for simclr_aug in simclr_augs:
                
                folders = [f"../weights/SimCLR_{encoder}_{image_size}_{simclr_aug}/checkpoint_{i}.ckpt" for i in range(1)]

                for rep, exp_folder in enumerate(folders):

                    # Checkpoint path
                    checkpoint_path = os.path.join(exp_folder)

                    # Load weights
                    simclr = SimCLR.load_from_checkpoint(checkpoint_path)

                    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_simclr_classifier(
                        simclr, 
                        augmentation,
                        rep,
                        p,
                        with_features,
                        paths[p])

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
                title = f"""Confusion matrix linear classifier (labels {p}%)
({label_features[with_features][0]}, {label_aug2[simclr.augmentation][0]}, {label_aug2[augmentation][0]}, {label_encoder[encoder]})
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
                file = f"figures/LC-{label_encoder[encoder]}-{p}-Validation-{label_features[with_features][1]}-{label_aug2[simclr.augmentation][1]}-{label_aug2[augmentation][1]}-{image_size}.png"
                plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)


                # Plot confusion matrix (test)
                # ----------------------------
                title = f"""Confusion matrix linear classifier (labels {p}%)
({label_features[with_features][0]}, {label_aug2[simclr.augmentation][0]}, {label_aug2[augmentation][0]}, {label_encoder[encoder]})
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
                file = f"figures/LC-{label_encoder[encoder]}-{p}-Test-{label_features[with_features][1]}-{label_aug2[simclr.augmentation][1]}-{label_aug2[augmentation][1]}-{image_size}.png"
                plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
