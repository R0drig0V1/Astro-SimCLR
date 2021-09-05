import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

import torch
import torchvision
#import torchmetrics

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------

sys.path.append('utils')
import utils_dataset
from utils_dataset import img_float2int, one_hot_trans, Dataset_stamps
from utils_plots import plot_confusion_matrix
from utils_metrics import results
from utils_training import trainer_v1

# -----------------------------------------------------------------------------

sys.path.append('losses')
from losses import P_stamps_loss

# -----------------------------------------------------------------------------

sys.path.append('models')
from models import P_stamp_net

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------

# Hiperparameters
epochs = 10
batch_size = 64
size = (21, 21)
lr = 1e-3
beta = 0.5
drop_r = 0.5
num_workers = 4

# -----------------------------------------------------------------------------

# Transformation to load images, float image is converted to np.array(np.int8) and
# resized
trans_stamp_load = transforms.Compose([transforms.Lambda(img_float2int),
                                       transforms.ToPILImage(),
                                       transforms.Resize(size),
                                       transforms.ToTensor()])

# Data reading
training_data = Dataset_stamps(data,
                               'Train',
                               device = device,
                               transform=trans_stamp_load,
                               target_transform=one_hot_trans)

validation_data = Dataset_stamps(data,
                                 'Validation',
                                 device=device,
                                 transform=trans_stamp_load,
                                 target_transform=one_hot_trans)

test_data = Dataset_stamps(data,
                           'Test',
                           device=device,
                           transform=trans_stamp_load,
                           target_transform=one_hot_trans)

# Data loaders
train_dataloader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

validation_dataloader = DataLoader(validation_data, num_workers=num_workers)

test_dataloader = DataLoader(test_data,batch_size=100, num_workers=num_workers)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Network
net = P_stamp_net(drop_r)
net.to(device)
#
criterion = P_stamps_loss(batch_size, beta)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5)
#optimizer = optim.Adam(net.parameters(), lr=lr, betas=[0.5, 0.9])
optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.5, 0.9))

#
trainer_v1(net, epochs, train_dataloader, validation_dataloader, optimizer, criterion, device)
PATH = '../weights/p_stamp_net_paper_loss.pth'
torch.save(net.state_dict(), PATH)

net = P_stamp_net(drop_r)
net.load_state_dict(torch.load(PATH))


# -----------------------------------------------------------------------------

net.eval()

y_true, y_pred = results(net, test_dataloader, torch.device("cpu"))
confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4], normalize='true')
acc = accuracy_score(y_true, y_pred)

title = 'Average confusion matrix p-stamps (p-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
file = 'Figures/conf_mat_pstamps_loss.png'
plot_confusion_matrix(confusion_matrix, title, utils_dataset.label_names, file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

"""
net = P_stamp_sim_clr()
net.to(device)

#trainer_v1(net, epochs, train_dataloader, optimizer, criterion, device)
PATH = '../weights/p_stamp_net_paper_loss.pth'
#torch.save(net.state_dict(), PATH)

net = P_stamp_net(drop_r)
net.load_state_dict(torch.load(PATH))

# -----------------------------------------------------------------------------

net.eval()

y_true, y_pred = results(net, test_dataloader, torch.device("cpu"))
confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4], normalize='true')
acc = accuracy_score(y_true, y_pred)

title = 'Average confusion matrix p-stamps (p-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
file = 'Figures/conf_mat_pstamps_loss.png'
plot_confusion_matrix(confusion_matrix, title, utils_dataset.label_names, file)

"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
