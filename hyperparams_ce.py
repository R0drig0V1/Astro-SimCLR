import os
import ray
import warnings
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config, resources_per_trial
from utils.plots import plot_confusion_matrix_mean_std
from utils.repeater_lib import hyperparameter_columns, summary
from utils.repeater_lib import path_best_hyperparameters
from utils.training import Supervised_Cross_Entropy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune

from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest import Repeater
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

# -----------------------------------------------------------------------------

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

# Requirement for Ray
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# -----------------------------------------------------------------------------

class ModelCheckpoint_V2(ModelCheckpoint):

    def __init__(self, monitor, dirpath, filename, save_top_k, mode):
        super().__init__(monitor=monitor,
                         dirpath=dirpath,
                         filename=filename,
                         save_top_k=save_top_k,
                         mode=mode)

    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        return self.format_checkpoint_name(monitor_candidates)


# -----------------------------------------------------------------------------

def ce_trainer(config_hyper, max_epochs=100, gpus=1):

    """
    Train the Supervised Cross Entropy model with the hyperparameters in
    config_hyper.
    """

    # Model's hyperparameters
    image_size     = config_hyper['image_size']
    batch_size     = config_hyper['batch_size']
    drop_rate      = config_hyper['drop_rate']
    beta_loss      = config_hyper['beta_loss']
    lr             = config_hyper['lr']
    optimizer      = config_hyper['optimizer']
    with_features  = config_hyper['with_features']
    balanced_batch = config_hyper['balanced_batch']
    augmentation   = config_hyper['augmentation']


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
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_val",
        dirpath=".",
        filename="checkpoint",
        save_top_k=1,
        mode="max"
    )


    # Logger for results
    tb_logger = TensorBoardLogger(
        save_dir=".",
        version=".",
        name="."
    )
 

    # Target for ray_tune
    tune_report = TuneReportCallback(
        metrics={"accuracy": "accuracy_val"},
        on="validation_end"
    )


    # Initialize pytorch_lightning module
    model = Supervised_Cross_Entropy(
        image_size=image_size,
        batch_size=batch_size,
        drop_rate=drop_rate,
        beta_loss=beta_loss,
        lr=lr,
        optimizer=optimizer,
        with_features=with_features,
        balanced_batch=balanced_batch,
        augmentation=augmentation
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            tune_report
        ],
        logger=tb_logger,
        progress_bar_refresh_rate=False,
        weights_summary=None
    )


    # Training
    trainer.fit(model)

    return None

# -----------------------------------------------------------------------------

# It generates the name for a trial
def trial_name_creator(trial):
    return "id_{}".format(trial.trial_id)

# -----------------------------------------------------------------------------

def ce_tune(num_samples, max_epochs, cpus_per_trial=1, gpus_per_trial=1):

    # Hyperparameters of model
    config_hyper = {
        "image_size": tune.choice([27]),
        "batch_size": tune.choice([15, 30, 45, 60, 75]),
        "drop_rate": tune.choice([0.2, 0.5, 0.8]),
        "beta_loss": tune.loguniform(1e-4, 1),
        "lr": tune.loguniform(5e-5, 5e-3),
        "optimizer": tune.choice(['AdamW', 'SGD']),
        "with_features": tune.choice([True]),
        "balanced_batch": tune.choice([False]),
        "augmentation": tune.choice(['without_aug'])
    }


    # Name of hyperparameters
    parameter_columns = [
        "image_size",
        "batch_size",
        "drop_rate",
        "beta_loss",
        "lr",
        "optimizer"
        ]


    # Scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=max_epochs,
        reduction_factor=2
    )


    # Report progress
    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=["accuracy", "training_iteration"]
    )


    # Function with trainer and parameters
    params = tune.with_parameters(
        ce_trainer,
        max_epochs=max_epochs,
        gpus=gpus_per_trial
    )


    # csv and json results
    tune_csv_logger = CSVLoggerCallback()
    tune_json_logger = JsonLoggerCallback()

    
    current_best_params = [{
        "image_size": 27,
        "batch_size": 75,
        "drop_rate": 0.2,
        "beta_loss": 0.5,
        "lr": 0.001,
        "optimizer": 'AdamW',
        "with_features": True,
        "balanced_batch": False,
        "augmentation": 'without_aug'
        }]

    # Searcher
    search_alg = HyperOptSearch(
        metric="accuracy",
        mode="max",
        points_to_evaluate=current_best_params)


    # Repeater to train with different seeds
    repeat = 5
    re_search_alg = Repeater(search_alg, repeat=repeat, set_index=True)


    # Hyperparameters tunning
    analysis = tune.run(
        params,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        metric="accuracy",
        mode="max",
        config=config_hyper,
        num_samples=num_samples*repeat,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="../hyperparameter_tuning",
        name="CE",
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_name_creator,
        verbose=1,
        max_failures=2,
        callbacks=[tune_csv_logger, tune_json_logger],
        search_alg=re_search_alg
    )


    # Dataframe with results
    df = analysis.dataframe(metric="accuracy", mode="max")

    # Dataframe is saved
    df.to_csv('results/hyperparameter_tuning_ce.csv', float_format='%.6f')

    # Summary of hyperparameter tuning
    df_summary = summary(df, metric="accuracy")
    df_summary.to_csv('results/summary_tuning_ce.csv', float_format='%.6f')

    # Checkpoints of the best hyperparameter combination
    checkpoints = path_best_hyperparameters(df)

    return checkpoints

# -----------------------------------------------------------------------------

# Hyperparameters tunning
checkpoints = ce_tune(
    num_samples=20,
    max_epochs=220,
    cpus_per_trial=resources_per_trial.cpus,
    gpus_per_trial=resources_per_trial.gpus
    )

# -----------------------------------------------------------------------------

# Save accuracies and confusion matrixes for different initial conditions
acc_array_val = []
conf_mat_array_val = []
acc_array_test = []
conf_mat_array_test = []


# Train for different initial conditions
for checkpoint_path in checkpoints:

    # Load weights
    sce = Supervised_Cross_Entropy.load_from_checkpoint(checkpoint_path)

    # Load dataset and computes confusion matrixes
    sce.prepare_data()

    # Compute metrics
    acc_test, conf_mat_test = sce.confusion_matrix(dataset='Test')
    acc_val, conf_mat_val = sce.confusion_matrix(dataset='Validation')

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

# -----------------------------------------------------------------------------

# Plot confusion matrix (validation)
title = f'Confusion matrix Stamps classifier\n Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}'
file = 'figures/confusion_matrix_CE_Validation.png'
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
title = f'Confusion matrix Stamps classifier\n Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}'
file = 'figures/confusion_matrix_CE_Test.png'
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Dataframe with results
df = pd.read_csv('results/hyperparameter_tuning_ce.csv', index_col=0)

# Summary of hyperparameter tuning
df_summary = pd.read_csv('results/summary_tuning_ce.csv', index_col=0)

# Names of hyperparameter columns of dataframe
hyper_columns = hyperparameter_columns(df)

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

# Copy the best hyperparameters's set
shutil.copy(hparams_path, "results/hparams_best_ce.yaml")

# -----------------------------------------------------------------------------
