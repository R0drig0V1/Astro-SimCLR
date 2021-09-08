import os
import sys
import pickle
import torch
import torchvision
import torchmetrics

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------

sys.path.append('utils')
import utils_dataset
from utils import Args
from utils_dataset import Dataset_stamps
from utils_plots import plot_confusion_matrix
from utils_metrics import results
from utils_training import Supervised_Cross_Entropy


# -----------------------------------------------------------------------------

sys.path.append('losses')
from losses import P_stamps_loss

# -----------------------------------------------------------------------------

sys.path.append('models')
from models import P_stamps_net

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

# Hiperparameters
args = Args({


# distributed training
'num_nodes': 1,
'gpus': 1,
'workers': 4,
'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),


# train options
'batch_size': 64,
'image_size': 21,
'max_epochs': 100,

# model options
'drop_ratio': 0.5,

# loss options
'optimizer': "SGD",
'lr': 1e-3,
'beta': 0.5,


# reload options
'model_path': "../weights"})


# -----------------------------------------------------------------------------

# Training 
sce = Supervised_Cross_Entropy(args)
trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, benchmark=True)
trainer.fit(sce)

# Computation of performance metrics for test and validation sets
val_results = trainer.validate(sce, verbose=False)
test_results = trainer.test(sce, verbose=False)

# Prints performance metrics for validation set 
print('\n\n', 'Validation results\n', '-'*80)
for key, value in val_results[0].items():
    print("{0:<10}:{1:.3f}".format(key, value))

# Prints performance metrics for test set 
print('\n\n', 'Test results\n', '-'*80)
for key, value in test_results[0].items():
    print("{0:<10}:{1:.3f}".format(key, value))

# Save weights
file = "P_stamps_epoch{0}.tar".format(args.max_epochs)
path = os.path.join(args.model_path, file)
torch.save(sce.model.state_dict(), path)

# Inicialize network
#model = P_stamps_net(args.drop_ratio, n_features=5, last_act_function='Softmax')
#model.load_state_dict(torch.load(path, map_location=args.device.type))

# -----------------------------------------------------------------------------