import os
import ray
import shutil
import warnings
import yaml

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from utils.config import config, resources_per_trial
from utils.plots import plot_confusion_matrix_mean_std
from utils.repeater_lib import hyperparameter_columns, summary
from utils.repeater_lib import path_best_hyperparameters

from utils.training import SimCLR

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from box import Box

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

# Load the hyperparamters of the model
file = open("results/hparams_best_ce.yaml", 'r')
hparams = Box(yaml.load(file, Loader=yaml.FullLoader))

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

def simclr_trainer(config_hyper, max_epochs=100, gpus=1):

    # Model's hyperparameters
    encoder_name          = config_hyper['encoder_name']
    method                = config_hyper['method']
    image_size            = config_hyper['image_size']
    augmentation          = config_hyper['augmentation']
    projection_dim        = config_hyper['projection_dim']
    temperature           = config_hyper['temperature']
    lr_encoder            = config_hyper['lr_encoder']
    batch_size_encoder    = config_hyper['batch_size_encoder']
    optimizer_encoder     = config_hyper['optimizer_encoder']
    beta_loss             = config_hyper['beta_loss']
    lr_classifier         = config_hyper['lr_classifier']
    batch_size_classifier = config_hyper['batch_size_classifier']
    optimizer_classifier  = config_hyper['optimizer_classifier']
    drop_rate_classifier  = config_hyper['drop_rate_classifier']
    with_features         = config_hyper['with_features']


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.001,
        patience=70,
        mode="max",
        check_finite=True
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
        metrics={
            "accuracy": "accuracy_val",
            "loss_enc": "loss_enc_val"},
        on="validation_end"
    )


    # Initialize pytorch_lightning module
    model = SimCLR(
        encoder_name=encoder_name,
        method=method,
        image_size=image_size,
        augmentation=augmentation,
        projection_dim=projection_dim,
        temperature=temperature,
        lr_encoder=lr_encoder,
        batch_size_encoder=batch_size_encoder,
        optimizer_encoder=optimizer_encoder,
        beta_loss=beta_loss,
        lr_classifier=lr_classifier,
        batch_size_classifier=batch_size_classifier,
        optimizer_classifier=optimizer_classifier,
        drop_rate_classifier=drop_rate_classifier,
        with_features=with_features
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

def simclr_tune(num_samples, max_epochs, cpus_per_trial=1, gpus_per_trial=1):

    # Hyperparameters of model
    config_hyper = {
    'encoder_name': tune.choice(['pstamps']),
    'method': tune.choice(['simclr']),
    'image_size': tune.choice([hparams.image_size]),
    'augmentation': tune.choice(['simclr']),
    'projection_dim': tune.choice([300]),
    'temperature': tune.loguniform(0.05, 0.15),
    'lr_encoder': tune.loguniform(0.05, 1.0),
    'batch_size_encoder': tune.choice([550]),
    'optimizer_encoder': tune.choice(['LARS']),
    'beta_loss': tune.choice([hparams.beta_loss]),
    'lr_classifier': tune.choice([hparams.lr]),
    'batch_size_classifier': tune.choice([hparams.batch_size]),
    'optimizer_classifier':  tune.choice([hparams.optimizer]),
    'drop_rate_classifier': tune.choice([hparams.drop_rate]),
    'with_features': tune.choice([False])
    }


    # Name of hyperparameters
    parameter_columns = {
        'encoder_name': 'enc',
        'method': 'method',
        'image_size': 'img',
        'augmentation': 'aug',
        'projection_dim': 'proj_dim',
        'temperature': 'temp',
        'lr_encoder': 'lr_enc',
        'batch_size_encoder': 'batch_enc',
        'optimizer_encoder': 'opt_enc'
        }


    # Scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=max_epochs,
        reduction_factor=2
    )


    # Report progress
    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns={
            "accuracy": "acc",
            "loss_enc": "loss_enc",
            "training_iteration": "iter"
        }
    )


    # Function with trainer and parameters
    params = tune.with_parameters(
        simclr_trainer,
        max_epochs=max_epochs,
        gpus=gpus_per_trial
    )


    # csv and json results
    tune_csv_logger = CSVLoggerCallback()
    tune_json_logger = JsonLoggerCallback()

    
    # Searcher
    search_alg = HyperOptSearch(metric="accuracy", mode="max")


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
        name="SimCLR",
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_name_creator,
        verbose=1,
        max_failures=2,
        raise_on_failed_trial=False, 
        callbacks=[tune_csv_logger, tune_json_logger],
        search_alg=re_search_alg#,
        #resume=True
    )


    # Dataframe with results
    df = analysis.dataframe(metric="accuracy", mode="max")

    # Dataframe is saved
    df.to_csv('results/hyperparameter_tuning_simclr.csv', float_format='%.6f')

    # Summary of hyperparameter tuning
    df_summary = summary(df, metric="accuracy")
    df_summary.to_csv('results/summary_tuning_simclr.csv', float_format='%.6f')

    # Checkpoints of the best hyperparameter combination
    checkpoints = path_best_hyperparameters(df)

    return checkpoints

# -----------------------------------------------------------------------------

# Hyperparameters tunning
checkpoints = simclr_tune(
    num_samples=15,
    max_epochs=850,
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
    simclr = SimCLR.load_from_checkpoint(checkpoint_path)

    # Load dataset and computes confusion matrixes
    simclr.prepare_data()

    # Compute metrics
    acc_test, conf_mat_test = simclr.confusion_matrix(dataset='Test')
    acc_val, conf_mat_val = simclr.confusion_matrix(dataset='Validation')

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
title = f"""Confusion matrix SimCLR classifier
(Without features, Simclr-aug, Stamps)
Accuracy Validation:{acc_mean_val:.3f}$\pm${acc_std_val:.3f}"""
file = f"figures/confusion_matrix_SimCLR-Validation-without_features-simclr_aug.png"
plot_confusion_matrix_mean_std(conf_mat_mean_val, conf_mat_std_val, title, file)

# Plot confusion matrix (test)
title = f"""Confusion matrix SimCLR classifier
(Without features, Simclr-aug, Stamps)
Accuracy Test:{acc_mean_test:.3f}$\pm${acc_std_test:.3f}"""
file = f"figures/confusion_matrix_SimCLR-Test-without_features-simclr_aug.png"
plot_confusion_matrix_mean_std(conf_mat_mean_test, conf_mat_std_test, title, file)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Dataframe with results
df = pd.read_csv('results/hyperparameter_tuning_simclr.csv', index_col=0)

# Summary of hyperparameter tuning
df_summary = pd.read_csv('results/summary_tuning_simclr.csv', index_col=0)

# Names of hyperparameter columns of dataframe
hyper_columns = hyperparameter_columns(df)

# -----------------------------------------------------------------------------

# Dataframe of the best hyperparameters for each combination in hyper
df_best_clr = df_summary.loc[df_summary['mean'].idxmax()].reset_index(drop=True)

# Mask of the best hyperparamter combination
where = 1
for hyper_name, opt_hyper in zip(hyper_columns, df_best_clr):
    where = where & (df[hyper_name] == opt_hyper)

# Folders of the best hyperparameter combination
folder = list(df['logdir'][where])[0]

# Load the hyperparamters of the model
hparams_path = os.path.join(folder, "hparams.yaml")

# Copy the best hyperparameters's set
shutil.copy(hparams_path, "results/hparams_best_simclr.yaml")

# -----------------------------------------------------------------------------
