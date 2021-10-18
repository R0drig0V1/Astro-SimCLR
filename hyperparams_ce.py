import os
import ray
import warnings

import numpy as np
import pytorch_lightning as pl

from utils.args import Args
from utils.config import config, resources_per_trial
from utils.plots import plot_confusion_matrix_mean_std
from utils.repeater import hyperparameter_columns, summary
from utils.repeater import folder_best_hyperparameters

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

# Last version of pytorch is unstable
#warnings.filterwarnings("ignore")

# Sets seed
#pl.utilities.seed.seed_everything(seed=1, workers=False)

# Requirement for Ray
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'

# -----------------------------------------------------------------------------

def train_mnist_tune(config_hyper, max_epochs=100, gpus=1):

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
    balanced_batch = config_hyper['balanced_batch']
    optimizer      = config_hyper['optimizer']


    # Early stop criterion
    early_stop_callback = EarlyStopping(
        monitor="accuracy_val",
        min_delta=0.001,
        patience=40,
        mode="max",
        check_finite=True,
        divergence_threshold=0.3
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


    # Save checkpoint
    tune_checkpoint = TuneReportCheckpointCallback(
        metrics={"accuracy": "accuracy_val"},
        filename="checkpoint.ckpt",
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
        balanced_batch=balanced_batch
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback,
            tune_report,
            tune_checkpoint
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


def tune_mnist_asha(num_samples=10, max_epochs=100, cpus_per_trial=1, gpus_per_trial=1):

    # Hyperparameters of model
    config_hyper = {
        "image_size": tune.choice([21]),
        "batch_size": tune.choice([35, 70, 140]),
        "drop_rate": tune.choice([0.25, 0.5, 0.75]),
        "beta_loss": tune.loguniform(1e-4, 1e+1),
        "lr": tune.loguniform(1e-4, 1e-1),
        "balanced_batch": tune.choice([True, False]),
        "optimizer": tune.choice(['AdamW', 'SGD', 'LARS'])
    }


    # Name of hyperparameters
    parameter_columns = [
        "image_size",
        "batch_size",
        "lr",
        "drop_rate",
        "beta_loss",
        "balanced_batch",
        "optimizer"]


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
        train_mnist_tune,
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
        keep_checkpoints_num=1,
        checkpoint_score_attr="accuracy",
        config=config_hyper,
        num_samples=num_samples*repeat,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="../hyperparameter_tuning",
        name="CE",
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_name_creator,
        verbose=1,
        callbacks=[tune_csv_logger, tune_json_logger],
        search_alg=re_search_alg
    )


    # Dataframe with results
    df = analysis.dataframe(metric="accuracy", mode="max")

    # Dataframe is saved
    df.to_csv('results/hyperparameter_tuning_ce.csv', float_format='%.6f')

    # Summary of hyperparameter tuning
    df_summary = summary(df)
    df_summary.to_csv('results/summary_tuning_ce.csv', float_format='%.6f')

    # Checkpoints of the best hyperparameter combination
    folders = folder_best_hyperparameters(df)

    return folders

# -----------------------------------------------------------------------------

# Hyperparameters tunning
folders = tune_mnist_asha(
    num_samples=30,
    max_epochs=250,
    cpus_per_trial=resources_per_trial.cpus,
    gpus_per_trial=resources_per_trial.gpus
    )

# -----------------------------------------------------------------------------

# Save accuracies and confusion matrixes for different initial conditions
acc_array = []
conf_mat_array = []

# Train for different initial conditions
for checkpoint_path in folders:

    # Load weights
    sce = Supervised_Cross_Entropy.load_from_checkpoint(checkpoint_path)

    # Load dataset and computes confusion matrixes
    sce.prepare_data()

    # Compute metrics
    acc, conf_mat = sce.confusion_matrix(dataset='Test')

    # Save metrics
    acc_array.append(acc)
    conf_mat_array.append(conf_mat)


# Compute mean and standard deviation of accuracy and confusion matrix
acc_mean = np.mean(acc_array, axis=0)
acc_std = np.std(acc_array, axis=0)
conf_mat_mean = np.mean(conf_mat_array, axis=0)
conf_mat_std = np.std(conf_mat_array, axis=0)


# Plot confusion matrix
title = f'Confusion matrix P-stamps (P-stamps loss)\n Accuracy Test:{acc_mean:.3f}$\pm${acc_std:.3f}'
file = 'Figures/confusion_matrix_CE_Test.png'
plot_confusion_matrix_mean_std(conf_mat_mean, conf_mat_std, title, file)

# -----------------------------------------------------------------------------
