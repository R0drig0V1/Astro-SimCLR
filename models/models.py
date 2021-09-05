import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamp_net(nn.Module):

    def __init__(self, drop_r):
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
        self.bn1 = nn.BatchNorm1d(90)
        self.drop = nn.Dropout(p=drop_r)
        self.fc3 = nn.Linear(90, 64)
        self.fc4 = nn.Linear(64,5)

    def forward(self, x1, x2):

        """
        x1: images
        x2: features
        """

        x1 = self.zpad(x1)

        r1 = self.conv(torch.rot90(x1, 0, [2, 3]))
        r2 = self.conv(torch.rot90(x1, 1, [2, 3]))
        r3 = self.conv(torch.rot90(x1, 2, [2, 3]))
        r4 = self.conv(torch.rot90(x1, 3, [2, 3]))

        x1 = (r1 + r2 + r3 + r4) / 4

        x = torch.cat((x1, x2), dim=1)

        x = self.bn1(x)
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x

    def conv(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.fl1(x)
        x = F.relu(self.fc1(x))

        return x

# -----------------------------------------------------------------------------