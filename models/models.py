import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

# Github: Spijkervet
class Linear_classifier(nn.Module):

    def __init__(self, input_size, n_classes):
        super(Linear_classifier, self).__init__()

        self.model = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

# -----------------------------------------------------------------------------

class Three_layers_classifier(nn.Module):

    def __init__(self, input_size_f1, input_size_f2, input_size_f3, n_classes, drop_rate):
        super().__init__()

        self.drop = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(input_size_f1, input_size_f2)
        self.fc2 = nn.Linear(input_size_f2, input_size_f3)
        self.fc3 = nn.Linear(input_size_f3, n_classes)

    def forward(self, x):

        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.softmax(self.fc3(x), dim=1)

        return y

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

        self.with_features = with_features

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=3)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=3)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=3)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(5184, 64, bias=False)

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
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x

# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamps_net_SimCLR(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=3)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=3)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=3)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(5184, 64, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
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
