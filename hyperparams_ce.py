import os
import pickle
import sys
import ray
import warnings

import pytorch_lightning as pl

from utils.args import Args
from utils.config import config, resources_per_trial
from utils.training import Supervised_Cross_Entropy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

# -----------------------------------------------------------------------------

# Last version of pytorch is unstable
#warnings.filterwarnings("ignore")

# Sets seed
pl.utilities.seed.seed_everything(seed=1, workers=False)

# Requirement for Ray
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

def train_mnist_tune(config_hyper, max_epochs=100, gpus=1):

    """
    It trains the Supervised Cross Entropy model with the hyperparameters in
    config_hyper.
    """

    # Model's hyperparameters
    image_size = 21
    batch_size = config_hyper['batch_size']
    drop_rate = config_hyper['drop_rate']
    beta_loss = config_hyper['beta_loss']
    lr = config_hyper['lr']
    balanced_batch = config_hyper['balanced_batch']
    optimizer = config_hyper['optimizer']

    # Inicializes pytorch_lightning module
    model= Supervised_Cross_Entropy(image_size=image_size,
                                    batch_size=batch_size,
                                    drop_rate=drop_rate,
                                    beta_loss=beta_loss,
                                    lr=lr,
                                    optimizer=optimizer,
                                    balanced_batch=balanced_batch)

    # Logger for results
    logger = TensorBoardLogger("tb_logs", name="")
 
    # Target for ray_tune
    tune_callback = TuneReportCallback({"accuracy": "Accuracy"}, on="validation_end")

    # Early stop criterion
    early_stop_callback = EarlyStopping(monitor="Accuracy",
                                        min_delta=0.002,
                                        patience=40,
                                        mode="max",
                                        check_finite=True,
                                        divergence_threshold=0.3)

    # Save checkpoint
    tune_checkpoint = TuneReportCheckpointCallback(
        metrics={"accuracy": "Accuracy"},
        filename="trainer.ckpt",
        on="validation_end")


    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback,
            tune_callback,
            tune_checkpoint
        ],
        logger=logger,
        progress_bar_refresh_rate=False,
        weights_summary=None)

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
        "batch_size": tune.choice([35, 70, 140]),
        "drop_rate": tune.choice([0.25, 0.5, 0.75]),
        "beta_loss": tune.loguniform(1e-4, 1e+1),
        "lr": tune.loguniform(1e-4, 1e-1),
        "balanced_batch": tune.choice([True, False]),
        "optimizer": tune.choice(['AdamW', 'SGD'])}

    # Scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=max_epochs,
        reduction_factor=2)

    # Name of hyperparameters
    parameter_columns = [
        "batch_size",
        "lr",
        "drop_rate",
        "beta_loss",
        "balanced_batch",
        "optimizer"]

    # It reports progress
    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=["accuracy", "training_iteration"])

    # Function with trainer and parameters
    params = tune.with_parameters(
        train_mnist_tune,
        max_epochs=max_epochs,
        gpus=gpus_per_trial)


    # It explores hyperparameters
    analysis = tune.run(params,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        metric="accuracy",
        mode="max",
        keep_checkpoints_num=1,
        checkpoint_score_attr="accuracy",
        config=config_hyper,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="../hyperparams",
        name="Supervised_Cross_Entropy",
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_name_creator,
        verbose=1)

    # Best config based on accuracy
    best_config = analysis.get_best_config(
        metric="accuracy",
        mode="max",
        scope="all")

    # Dataframe with results
    df = analysis.dataframe(
        metric="accuracy",
        mode="max")

    # Dataframe is saved
    df.to_csv('results/hyper_supervised_cross_entropy.csv', float_format='%.6f')

    # Best hyperparameters are printed
    print("Best hyperparameters found were: ", best_config)

    # Path best checkpoint
    best_checkpoint_path = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("accuracy", "max", "all"),
        metric="accuracy",
        mode="max")

    return best_config, best_checkpoint_path

# -----------------------------------------------------------------------------

# Exploration of hyperparameters
hyperparams, path = tune_mnist_asha(
    num_samples=20,
    max_epochs=250,
    cpus_per_trial=resources_per_trial.cpus,
    gpus_per_trial=resources_per_trial.gpus)

# -----------------------------------------------------------------------------

# Loads weights
sce = Supervised_Cross_Entropy.load_from_checkpoint(path + "/trainer.ckpt")

# Load dataset and computes confusion matrixes
sce.prepare_data()
sce.conf_mat_val()
sce.conf_mat_test()

# -----------------------------------------------------------------------------
