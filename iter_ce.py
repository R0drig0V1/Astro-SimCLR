import os

import numpy as np
import pytorch_lightning as pl

from utils.config import config
from utils.plots import plot_confusion_matrix_mean_std
from utils.training import Supervised_Cross_Entropy

from box import Box
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# -----------------------------------------------------------------------------

gpus = [3]

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
        min_delta=0.001,
        patience=70,
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
        max_epochs=350,
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

# with_features   : True
# balanced_batch  : False
# augmentation    : False

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

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
(with features, without augmentations)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-with_features-without_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(with features, without augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-with_features-without_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)


# -----------------------------------------------------------------------------

# with_features   : False
# balanced_batch  : False
# augmentation    : False

# -----------------------------------------------------------------------------

# Hyperparameters
args["with_features"] = False
args["augmentation"] = False       

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-without_features-without_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(without features, without augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-without_features-without_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------

# with_features   : True
# balanced_batch  : False
# augmentation    : 'default'

# -----------------------------------------------------------------------------

# Hyperparameters
args["with_features"] = True
args["augmentation"] = 'default'

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
(with features, SimCLR augmentations)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-with_features-SimCLR_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(with features, SimCLR augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-with_features-SimCLR_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------

# with_features   : False
# balanced_batch  : False
# augmentation    : 'default'

# -----------------------------------------------------------------------------

# Hyperparameters
args["with_features"] = False
args["augmentation"] = 'default'

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
(without features, SimCLR augmentations)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-without_features-SimCLR_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(without features, SimCLR augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-without_features-SimCLR_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------

# with_features   : True
# balanced_batch  : False
# augmentation    : 'astro'

# -----------------------------------------------------------------------------

args["with_features"] = True
args["augmentation"] = 'astro'

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
(with features, Astro augmentations)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-with_features-Astro_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(with features, Astro augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-with_features-Astro_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------

# with_features   : False
# balanced_batch  : False
# augmentation    : 'astro'

# -----------------------------------------------------------------------------

# Hyperparameters
args["with_features"] = False
args["augmentation"] = 'astro'

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []

# Train for different initial conditions
for _ in range(5):

    # Train and compute metrics
    (acc_val, conf_mat_val), (acc_test, conf_mat_test) = training_ce(args)

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
(without features, Astro augmentations)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = 'Figures/confusion_matrix_CE-Validation-without_features-Astro_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
# ----------------------------
title = f"""Confusion matrix Stamps classifier
(without features, Astro augmentations)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = 'Figures/confusion_matrix_CE-Test-without_features-Astro_augmentation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
