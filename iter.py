import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

# -----------------------------------------------------------------------------

sys.path.append('utils')
import utils_dataset
from utils_dataset import dataset_structure
from utils_dataset import resize

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

# Class to load stamps
class Dataset_stamps(Dataset):

    def __init__(self, pickle, dataset, transform=None, target_transform=None):

        self.pickle = pickle
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pickle[self.dataset]['labels'])

    def __getitem__(self, idx):

        image = self.pickle[self.dataset]['images'][idx]
        label = self.pickle[self.dataset]['labels'][idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

epochs = 100
batch_size = 64
size = (21, 21)
lr = 1e-3
beta = 0.5
drop_r = 0.5

# -----------------------------------------------------------------------------

# Transformation to load image, np.array(np.int8) is converted to tensor

trans_stamp_load = transforms.Compose([transforms.Lambda(resize),
                                       transforms.ToPILImage(),
                                       transforms.Resize(size),
                                       transforms.ToTensor()
                                       ])

# One hot encoding to use softmax
trans_labels = None #lambda x: F.one_hot(torch.tensor(x), num_classes=5) 

# Loads data
training_data = Dataset_stamps(data, 'Train', transform=trans_stamp_load, target_transform=trans_labels)
validation_data = Dataset_stamps(data, 'Validation', transform=trans_stamp_load, target_transform=trans_labels)
test_data = Dataset_stamps(data, 'Test', transform=trans_stamp_load, target_transform=trans_labels)

# Data loader
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------

# P-stamp implementation
class Net_p_stamp(nn.Module):

    def __init__(self, size, drop_r):
        super().__init__()
        self.zpad = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(2304, 64)
        self.drop = nn.Dropout(p=drop_r)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64,5)

    def forward(self, x):

        x = self.zpad(x)
        #print(x.size())

        x1 = self.conv(torch.rot90(x, 0, [2, 3]))
        x2 = self.conv(torch.rot90(x, 1, [2, 3]))
        x3 = self.conv(torch.rot90(x, 2, [2, 3]))
        x4 = self.conv(torch.rot90(x, 3, [2, 3]))
        #x = self.conv(x)
        x = (x1 + x2 + x3 + x4) / 4

        x = self.drop(x)
        #print(x.size())
        x = F.relu(self.fc3(x))
        #print(x.size())
        #x = F.softmax(self.fc4(x), dim=0)
        x = self.fc4(x)
        #print(x.size())
        return x

    def conv(self, x):

        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.pool1(x)
        #print(x.size())
        x = F.relu(self.conv3(x))
        #print(x.size())
        x = F.relu(self.conv4(x))
        #print(x.size())
        x = F.relu(self.conv5(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        x = self.fl1(x)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        return x

# Network
net = Net_p_stamp(size, drop_r)

# -----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

for ep in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (ep + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
