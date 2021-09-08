import os
import sys
import pickle
import torch
import torchvision
import torchmetrics

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

sys.path.append('utils')
import utils_dataset
from utils import Args
from utils_dataset import Dataset_stamps
from utils_plots import plot_confusion_matrix
from utils_metrics import results
from utils_training import Self_Supervised_CLR
from utils_transformations import Augmentation_SimCLR, Resize_img

# -----------------------------------------------------------------------------

sys.path.append('losses')
from losses import P_stamps_loss, NT_Xent

# -----------------------------------------------------------------------------

sys.path.append('models')
from models import P_stamps_net, SimCLR_net, Linear_classifier

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

args = Args({

# distributed training
'num_nodes': 1,
'gpus': 1,
'workers': 4,
'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),


# train options
'batch_size': 128,
'image_size': 21,
'max_epochs': 100,


# model options
'projection_dim': 64,
'n_features': 5,
'drop_ratio': 0.5,
'n_classes': 5,


# loss options
'optimizer': "Adam",
'temperature': 0.5,
'lr': 1e-3,
'beta': 0.5,


# reload options
'model_path': "../weights",


# logistic regression options
'logistic_batch_size': 256,
'logistic_epochs': 500})


# -----------------------------------------------------------------------------

# Data reading
training_data_aug = Dataset_stamps(data,
                               'Train',
                               transform=Augmentation_SimCLR(size=args.image_size),
                               target_transform=None)

validation_data_aug = Dataset_stamps(data,
                                 'Validation',
                                 transform=Augmentation_SimCLR(size=args.image_size),
                                 target_transform=None)

# Data loaders
train_dataloader_aug = DataLoader(training_data_aug,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=True)

validation_dataloader_aug = DataLoader(validation_data_aug,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   drop_last=True)

# -----------------------------------------------------------------------------

# Data reading
training_data = Dataset_stamps(data,
                               'Train',
                               transform=Resize_img(size=args.image_size),
                               target_transform=None)

test_data = Dataset_stamps(data,
                           'Test',
                           transform=Resize_img(size=args.image_size),
                           target_transform=None)

validation_data_aug = Dataset_stamps(data,
                                 'Validation',
                                 transform=Resize_img(size=args.image_size),
                                 target_transform=None)

# Data loaders
train_dataloader = DataLoader(training_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=True)

validation_dataloader_aug = DataLoader(validation_data_aug,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   drop_last=True)

test_dataloader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.workers,
                             drop_last=True)

# -----------------------------------------------------------------------------

ssclr = Self_Supervised_CLR(args)
trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, benchmark=True)
trainer.fit(ssclr, train_dataloader_aug)


file = "SimCLR_epoch{0}.tar".format(args.max_epochs)
path = os.path.join(args.model_path, file)
torch.save(ssclr.model.state_dict(), path)

# -----------------------------------------------------------------------------


def inference(loader, simclr_model, device):

    feature_vector = []
    labels_vector = []

    for step, (x_img, x_feat, y) in enumerate(loader):

        x_img = x_img.to(device)
        x_feat = x_feat.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x_img, x_img, x_feat)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    print("Features shape {}".format(feature_vector.shape))

    return feature_vector, labels_vector





def get_features(simclr_model, train_loader, test_loader, device):

    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)

    return train_X, train_y, test_X, test_y




def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




def train(args, loader, simclr_model, model, criterion, optimizer):

    loss_epoch = 0
    accuracy_epoch = 0

    for step, (x, y) in enumerate(loader):

        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = torch.argmax(output, dim=1)
        true = torch.argmax(y, dim=1)

        acc = (predicted == true).sum().item() / true.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch




def test(args, loader, simclr_model, model, criterion, optimizer):

    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()

    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = torch.argmax(output, dim=1)
        true = torch.argmax(y, dim=1)

        acc = (predicted == true).sum().item() / true.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


# Initializes network
encoder = P_stamps_net(drop_ratio=args.drop_ratio, n_features=args.n_features, last_act_function="Identity")
simclr_model = SimCLR_net(encoder, args.projection_dim, args.n_features)

simclr_model.load_state_dict(torch.load(path, map_location=args.device.type))

simclr_model = simclr_model.to(args.device)
simclr_model.eval()


## Logistic Regression

linear_model = Linear_classifier(args.n_features, args.n_classes)
linear_model = linear_model.to(args.device)

optimizer = torch.optim.Adam(linear_model.parameters(), lr=3e-4)
criterion = P_stamps_loss(args.batch_size, args.beta)

print("### Creating features from pre-trained context model ###")
train_X, train_y, test_X, test_y = get_features(simclr_model, train_dataloader, test_dataloader, args.device)

arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.logistic_batch_size)

for epoch in range(args.logistic_epochs):

    loss_epoch, accuracy_epoch = train(args, arr_train_loader, simclr_model, linear_model, criterion, optimizer)
    print(f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}")


# final testing
loss_epoch, accuracy_epoch = test(args, arr_test_loader, simclr_model, linear_model, criterion, optimizer)
print(f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}")
