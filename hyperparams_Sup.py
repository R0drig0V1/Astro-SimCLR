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

    # Parameters of model
    #image_size = config_hyper['image_size']
    batch_size = config_hyper['batch_size']
    drop_rate = config_hyper['drop_rate']
    beta_loss = config_hyper['beta_loss']
    lr = config_hyper['lr']
    balanced_batch = config_hyper['balanced_batch']
    optimizer = config_hyper['optimizer']


    # Inicializes pytorch_lightning module
    model= Supervised_Cross_Entropy(image_size=21,
                                    batch_size=batch_size,
                                    drop_rate=drop_rate,
                                    beta_loss=beta_loss,
                                    lr=lr,
                                    optimizer=optimizer,
                                    balanced_batch=balanced_batch)


    logger = TensorBoardLogger('tb_logs')

 
    # Target for ray_tune
    tune_callback = TuneReportCallback({"accuracy": "Accuracy"}, on="validation_end")


    # Early stop criterion
    early_stop_callback = EarlyStopping(monitor="Accuracy",
                                        min_delta=0.002,
                                        patience=30,
                                        mode="max",
                                        check_finite=True,
                                        divergence_threshold=0.3,
                                        verbose=False)


    # Trainer
    trainer = pl.Trainer(max_epochs=max_epochs,
                         gpus=gpus,
                         benchmark=True,
                         callbacks=[early_stop_callback,tune_callback],
                         logger=logger,
                         progress_bar_refresh_rate=False,
                         weights_summary=None)

    trainer.fit(model)

    return None

# -----------------------------------------------------------------------------


#def trial_name_creator(trial):
#    return "{}".format(trial.trainable_name, trial.trial_id)

def trial_name_creator(trial):
    return "id_{}".format(trial.trial_id)



def tune_mnist_asha(num_samples=10, max_epochs=10, cpus_per_trial=1, gpus_per_trial=1):

    config_hyper = {"batch_size": tune.choice([35, 70, 140]),
                    "drop_rate": tune.choice([0.25, 0.5, 0.75]),
                    "beta_loss": tune.loguniform(1e-4, 1e+1),
                    "lr": tune.loguniform(1e-4, 1e-1),
                    "balanced_batch": tune.choice([True, False]),
                    "optimizer": tune.choice(['AdamW', 'SGD'])}

    scheduler = ASHAScheduler(max_t=max_epochs,
                              grace_period=max_epochs,
                              reduction_factor=2)

    parameter_columns = ["batch_size",
                         "lr",
                         "drop_rate",
                         "beta_loss",
                         "balanced_batch",
                         "optimizer"]


    reporter = CLIReporter(parameter_columns=parameter_columns,
                           metric_columns=["accuracy", "training_iteration"])

    params = tune.with_parameters(train_mnist_tune,
                                  max_epochs=max_epochs,
                                  gpus=gpus_per_trial)

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
                        name="Supervised_Cross_Entropy",
                        trial_name_creator=trial_name_creator,
                        trial_dirname_creator=trial_name_creator,
                        verbose=1)

    # Best config based on accuracy
    best_config = analysis.get_best_config(metric="accuracy", mode="max", scope="all")

    # Dataframe with results is saved
    df = analysis.dataframe(metric="accuracy", mode="max")
    df.to_csv('results/hyper_supervised_cross_entropy.csv', float_format='%.6f')

    # Best hyperparameters are printed
    print("Best hyperparameters found were: ", best_config)

    return best_config

# -----------------------------------------------------------------------------

# Hyperparameters exploration
hyperparams = tune_mnist_asha(num_samples=15,
                              max_epochs=250,
                              cpus_per_trial=resources_per_trial.cpus,
                              gpus_per_trial=resources_per_trial.gpus)

# -----------------------------------------------------------------------------

batch_size = hyperparams['batch_size']
drop_rate = hyperparams['drop_rate']
beta_loss = hyperparams['beta_loss']
lr = hyperparams['lr']
balanced_batch = hyperparams['balanced_batch']
optimizer = hyperparams['optimizer']

# Inicializes pytorch_lightning module
model = Supervised_Cross_Entropy(image_size=21,
                                 batch_size=batch_size,
                                 drop_rate=drop_rate,
                                 beta_loss=beta_loss,
                                 lr=lr,
                                 optimizer=optimizer,
                                 balanced_batch=balanced_batch)
 

# Save checkpoint
checkpoint_callback = ModelCheckpoint(monitor="Accuracy",
                                      dirpath=os.path.join(config.model_path),
                                      filename="best_hyper_supervised_cross_entropy",
                                      save_top_k=1,
                                      mode="max")

# Early stop criterion
early_stop_callback = EarlyStopping(monitor="Accuracy",
                                    min_delta=0.002,
                                    patience=30,
                                    mode="max",
                                    check_finite=True,
                                    divergence_threshold=0.3)

# Defining logger object
logger = TensorBoardLogger('tb_logs', name='Supervised_Cross_Entropy')

# Trainer
trainer = pl.Trainer(max_epochs=250,
                     gpus=1,
                     benchmark=True,
                     callbacks=[early_stop_callback,checkpoint_callback],
                     logger=logger)

trainer.fit(model)



# Path of best model
path = checkpoint_callback.best_model_path
print("\nBest model path:", path)

# Loads weights
sce = Supervised_Cross_Entropy.load_from_checkpoint(path)

# Load dataset and computes confusion matrixes
sce.prepare_data()
sce.conf_mat_val()
sce.conf_mat_test()

# -----------------------------------------------------------------------------
