import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor

# -----------------------------------------------------------------------------

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)

    return h, w


def maxpool_output_shape(h_w, kernel_size=1, stride=2):

    h = floor((h_w[0] - (kernel_size-1) - 1) / stride + 1)
    w = floor((h_w[1] - (kernel_size-1) - 1) / stride + 1)

    return h, w

# -----------------------------------------------------------------------------

# Github: Spijkervet
class Linear_classifier(nn.Module):

    def __init__(self, input_size, n_classes):
        super(Linear_classifier, self).__init__()

        self.model = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------

class Three_layers_classifier(nn.Module):

    def __init__(self, input_size_f1, input_size_f2, input_size_f3, n_classes, drop_rate):
        super().__init__()

        #self.bn1 = nn.BatchNorm1d(input_size_f1)
        self.fc1 = nn.Linear(input_size_f1, input_size_f2)
        self.drop = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(input_size_f2, input_size_f3)
        self.fc3 = nn.Linear(input_size_f3, n_classes)

    def forward(self, x):

        #x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        #y = F.softmax(self.fc3(x), dim=1)
        y = self.fc3(x)

        return y

# -----------------------------------------------------------------------------

class Three_layers_classifier2(nn.Module):

    def __init__(self, input_size_f1, input_size_f2, input_size_f3, n_classes, drop_rate):
        super().__init__()

        #self.bn1 = nn.BatchNorm1d(input_size_f1)
        self.fc1 = nn.Linear(input_size_f1, input_size_f2)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.bn = nn.BatchNorm1d(input_size_f2)
        self.fc2 = nn.Linear(input_size_f2, input_size_f3)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.fc3 = nn.Linear(input_size_f3, n_classes)

    def forward(self, x):

        #x = self.bn1(x)
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)
        x = self.bn(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc2(x)
        #x = self.drop2(x)
        #y = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)

        return x

# -----------------------------------------------------------------------------

# Github: Spijkervet
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamps_net(nn.Module):

    def __init__(self, drop_rate, with_features):
        super().__init__()

        kernel = 3

        if kernel == 3:
            padding = 1

        elif kernel == 5: 
            padding = 2

        elif kernel == 7:
            padding = 3

        self.with_features = with_features

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel, padding=padding)
        self.conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv5 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(2304, 64)

        if self.with_features:
            self.bn1 = nn.BatchNorm1d(87)
            self.fc2 = nn.Linear(87, 64)

        else:
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 64)

        self.drop = nn.Dropout(p=drop_rate)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 5)

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


    def forward(self, x_img, x_feat):

        """
        x_img: images
        x_feat: features
        """

        r1 = self.conv(torch.rot90(x_img, 0, [2, 3]))
        r2 = self.conv(torch.rot90(x_img, 1, [2, 3]))
        r3 = self.conv(torch.rot90(x_img, 2, [2, 3]))
        r4 = self.conv(torch.rot90(x_img, 3, [2, 3]))

        x_img = (r1 + r2 + r3 + r4) / 4

        if self.with_features: x = torch.cat((x_img, x_feat), dim=1)
        else: x = x_img

        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamps_net_SimCLR(nn.Module):

    def __init__(self):
        super().__init__()
        
        kernel = 3

        if kernel == 3:
            padding = 1

        elif kernel == 5: 
            padding = 2

        elif kernel == 7:
            padding = 3

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel, padding=padding)
        self.conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv5 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(2304, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)


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


    def forward(self, x_img):

        r1 = self.conv(torch.rot90(x_img, 0, [2, 3]))
        r2 = self.conv(torch.rot90(x_img, 1, [2, 3]))
        r3 = self.conv(torch.rot90(x_img, 2, [2, 3]))
        r4 = self.conv(torch.rot90(x_img, 3, [2, 3]))

        x = (r1 + r2 + r3 + r4) / 4

        x = self.bn1(x)
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))

        return x


# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamps_net_SimCLR_v2(nn.Module):

    def __init__(self, k=1):
        super().__init__()
        
        kernel = 3

        if kernel == 3:
            padding = 1

        elif kernel == 5: 
            padding = 2

        elif kernel == 7:
            padding = 3


        self.k = k
        self.c1 = int(k * 32)
        self.c2 = 2 * self.c1
        self.linear_output = int(k*128)


        self.conv1 = nn.Conv2d(3, self.c1, 4)
        output = conv_output_shape((27,27), kernel_size=4, pad=0)

        self.conv2 = nn.Conv2d(self.c1, self.c1, kernel, padding=padding)
        output = conv_output_shape(output, kernel_size=kernel, pad=padding)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        output = maxpool_output_shape(output, kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(self.c1, self.c2, kernel, padding=padding)
        output = conv_output_shape(output, kernel_size=kernel, pad=padding)

        self.conv4 = nn.Conv2d(self.c2, self.c2, kernel, padding=padding)
        output = conv_output_shape(output, kernel_size=kernel, pad=padding)

        self.conv5 = nn.Conv2d(self.c2, self.c2, kernel, padding=padding)
        output = conv_output_shape(output, kernel_size=kernel, pad=padding)

        self.pool2 = nn.MaxPool2d(2, stride=2)
        output = maxpool_output_shape(output, kernel_size=2, stride=2)

        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(self.c2*output[0]*output[1], self.linear_output, bias=False)
        self.bn1 = nn.BatchNorm1d(self.linear_output)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)


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


    def forward(self, x_img):

        r1 = self.conv(torch.rot90(x_img, 0, [2, 3]))
        r2 = self.conv(torch.rot90(x_img, 1, [2, 3]))
        r3 = self.conv(torch.rot90(x_img, 2, [2, 3]))
        r4 = self.conv(torch.rot90(x_img, 3, [2, 3]))

        x = (r1 + r2 + r3 + r4) / 4

        x = self.bn1(x)
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))

        return x

# -----------------------------------------------------------------------------

#  Spijkervet / SimCLR / simclr / simclr.py
class CLR(nn.Module):

    def __init__(self, encoder, input_size, projection_size):
        super(CLR, self).__init__()

        self.encoder = encoder

        # We use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(input_size, projection_size, bias=False),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size, bias=False),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size, bias=False),
            #nn.ReLU(),
            #nn.Linear(projection_size, projection_size, bias=False),
        )

    def forward(self, x_img_i, x_img_j, *x_feat):

        h_i = self.encoder(x_img_i, *x_feat)
        h_j = self.encoder(x_img_j, *x_feat)

        # Spherical normalization
        h_i = h_i * (1/torch.norm(h_i, dim=1, keepdim=True))
        h_j = h_j * (1/torch.norm(h_j, dim=1, keepdim=True))

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j

# -----------------------------------------------------------------------------
