import os
import pickle
import sys
import ray
import warnings

import pytorch_lightning as pl

from utils.args import Args
from utils.config import config, resources_per_trial
from utils.training import CLR_a, CLR_b

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

def train_mnist_tune(config_hyper, max_epochs=100, gpus=1, checkpoint_dir=''):

    """
    It trains the CLR model with the hyperparameters in config_hyper.
    """

    # Encoder's hyperparameters
    encoder = config_hyper["encoder"]

    # Dataset's hyperparameters
    image_size = config_hyper["image_size"]
    astro_augmentation = config_hyper["astro_augmentation"]
    batch_size = config_hyper["batch_size"]
    balanced_batch = config_hyper["balanced_batch"]

    # Optimizer's hyperparameters
    optimizer = config_hyper["optimizer"]
    lr = config_hyper["lr"]

    # Loss's hyperparameters
    temperature = config_hyper["temperature"]
    method = config_hyper["method"]

    # Proyection's hyperparameter
    projection_dim = 64

    # Classification with features
    with_features = config_hyper["with_features"]


    # Linear classifier's hyperparameters
    args_clr_b = Args({
        "batch_size": 100,
        "max_epochs": 100,
        "optimizer": "SGD",
        "lr": 1e-3,
        "beta_loss": 0.007725,
        "drop_rate": 0.25
    })


    # Logger for results
    logger = TensorBoardLogger(
        save_dir="simclr_a",
        name=""
    )
 

    # Save checkpoint
    checkpoint_callback_a = ModelCheckpoint(
        monitor="loss_val",
        #dirpath=checkpoint_dir,
        filename="clr_encoder{loss_val:.2f}'",
        save_top_k=1,
        mode="min"
    )


    # Early stop criterion
    early_stop_callback_a = EarlyStopping(
        monitor="loss_val",
        mode="min",
        patience=70,
        check_finite=True
    )


    # Initialize classifier
    clr_a = CLR_a(
        encoder_name=encoder,
        image_size=image_size,
        astro_augmentation=astro_augmentation,
        batch_size=batch_size,
        projection_dim=projection_dim,
        temperature=temperature,
        lr=lr,
        optimizer=optimizer,
        method=method
    )
 

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        gpus=gpus,
        benchmark=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback_a, early_stop_callback_a],
        logger=logger,
        progress_bar_refresh_rate=False,
        weights_summary=None
    )


    # Training
    trainer.fit(clr_a)

    # -------------------------------

    # Path of best model
    path = checkpoint_callback_a.best_model_path


    # Load weights
    clr_a = CLR_a.load_from_checkpoint(path)

    # -------------------------------


    # Save checkpoint
    tune_checkpoint = TuneReportCheckpointCallback(
        metrics={"accuracy": "accuracy_val"},
        filename="sim_b.ckpt",
        on="validation_end"
    )


   # Logger for results
    logger = TensorBoardLogger(
        save_dir="simclr_b",
        name=""
    )


    # Early stop criterion
    early_stop_callback_b = EarlyStopping(
        monitor="accuracy_val",
        mode="max",
        min_delta=0.002,
        patience=30,
        divergence_threshold=0.4,
        check_finite=True
    )


    # Target for ray_tune
    tune_callback = TuneReportCallback(
        metrics={"accuracy": "accuracy_val"},
        on="validation_end"
    )


    # Initialize classifier
    clr_b =CLR_b(
        clr_model=clr_a,
        image_size=image_size,
        batch_size=args_clr_b.batch_size,
        beta_loss=args_clr_b.beta_loss,
        lr=args_clr_b.lr,
        drop_rate=args_clr_b.drop_rate,
        optimizer=args_clr_b.optimizer,
        with_features=with_features
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,#args_clr_b.max_epochs,
        gpus=gpus,
        benchmark=True,
        callbacks=[
            early_stop_callback_b,
            tune_checkpoint,
            tune_callback],
        logger=logger,
        progress_bar_refresh_rate=False,
        weights_summary=None
    )


    # Training
    trainer.fit(clr_b)

    return None

# -----------------------------------------------------------------------------

# It generates the name for a trial
def trial_name_creator(trial):
    return "id_{}".format(trial.trial_id)

# -----------------------------------------------------------------------------


def tune_mnist_asha(num_samples=10, max_epochs=10, cpus_per_trial=1, gpus_per_trial=1):

    # Model's hyperparameters
    config_hyper = {
        "encoder": tune.choice(["pstamps", "resnet18", "resnet50"]),
        "image_size": tune.choice([21]),
        "astro_augmentation": tune.choice([True]),
        "batch_size": tune.choice([200, 350, 500]),
        "balanced_batch": tune.choice([True]),#, False]),
        "optimizer": tune.choice(['LARS']),
        "lr": tune.choice([100, 10, 1, 0.1, 0.01]),
        "temperature": tune.loguniform(1e-4, 1e+1),
        "method": tune.choice(["supcon", "simclr"]),
        "with_features": tune.choice([True]),# False])
    }


    # Name of hyperparameters
    parameter_columns = [
        "encoder",
        "image_size",
        "astro_augmentation"
        "batch_size",
        "balanced_batch",
        "optimizer",
        "lr",
        "temperature",
        "method",
        "with_features"
    ]


    # Scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=max_epochs,
        reduction_factor=2
    )


    # It reports progress
    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=["accuracy", "training_iteration"]
    )


    # Function with trainer and parameters
    params = tune.with_parameters(
        train_mnist_tune,
        max_epochs=max_epochs,
        gpus=gpus_per_trial#,
        #checkpoint_dir=checkpoint_dir
    )


    # hyperparameters tunning
    analysis = tune.run(
        params,
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
        name="CLR",
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_name_creator,
        verbose=1
    )


    # Best config based on accuracy
    best_config = analysis.get_best_config(
        metric="accuracy",
        mode="max",
        scope="all"
    )


    # Dataframe with results is saved
    df = analysis.dataframe(metric="accuracy", mode="max")

    # Dataframe is saved
    df.to_csv('results/hyperparameters_clr.csv', float_format='%.6f')

    # Best hyperparameters are printed
    print("Best hyperparameters found were: ", best_config)

    # Path best checkpoint
    best_checkpoint_path = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("accuracy", "max", "all"),
        metric="accuracy",
        mode="max"
    )

    return best_config, best_checkpoint_path

# -----------------------------------------------------------------------------

# Hyperparameters tunning
config_hyper, path = tune_mnist_asha(
    num_samples=20,
    max_epochs=100,
    cpus_per_trial=resources_per_trial.cpus,
    gpus_per_trial=resources_per_trial.gpus
)

# -----------------------------------------------------------------------------

# Encoder's hyperparameters
encoder = config_hyper["encoder"]

# Dataset's hyperparameters
image_size = config_hyper["image_size"]
astro_augmentation = config_hyper["astro_augmentation"]
batch_size = config_hyper["batch_size"]
balanced_batch = config_hyper["balanced_batch"]

# Optimizer's hyperparameters
optimizer = config_hyper["optimizer"]
lr = config_hyper["lr"]

# Loss's hyperparameters
temperature = config_hyper["temperature"]
method = config_hyper["method"]

# Proyection's hyperparameter
projection_dim = 64

# Classification with features
#with_features = config_hyper["with_features"]

# Initialize classifier
clr_a = CLR_a(
    encoder_name=encoder,
    image_size=image_size,
    astro_augmentation=astro_augmentation,
    batch_size=batch_size,
    projection_dim=projection_dim,
    temperature=temperature,
    lr=lr,
    optimizer=optimizer,
    method=method
)

# -----------------------------------------------------------------------------

# Load weights
clr_b = CLR_b.load_from_checkpoint(path + "/sim_b.ckpt", clr_model=clr_a)

# Load dataset and compute confusion matrixes
clr_b.prepare_data()
clr_b.conf_mat_val()
clr_b.conf_mat_test()

# -----------------------------------------------------------------------------

