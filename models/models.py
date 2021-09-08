import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

# P-stamp implementation
class P_stamps_net(nn.Module):

    def __init__(self, drop_ratio, n_features, last_act_function):
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
        self.drop = nn.Dropout(p=drop_ratio)
        self.fc3 = nn.Linear(90, 64)
        self.fc4 = nn.Linear(64, n_features)

        if(last_act_function == 'Softmax'):
            self.last_layer_func = nn.Softmax(dim=1)

        elif(last_act_function == 'ReLU'):
            self.last_layer_func = nn.ReLU()

        elif(last_act_function == 'Identity'):
            self.last_layer_func = nn.Identity()
        else:
            raise KeyError(f"Error: {last_layer_func} is not a valid activation function")


    def forward(self, x_img, x_feat):

        """
        x_img: images
        x_feat: features
        """

        x_img = self.zpad(x_img)

        r1 = self.conv(torch.rot90(x_img, 0, [2, 3]))
        r2 = self.conv(torch.rot90(x_img, 1, [2, 3]))
        r3 = self.conv(torch.rot90(x_img, 2, [2, 3]))
        r4 = self.conv(torch.rot90(x_img, 3, [2, 3]))

        x_img = (r1 + r2 + r3 + r4) / 4

        x = torch.cat((x_img, x_feat), dim=1)

        x = self.bn1(x)
        #x = F.relu(self.drop(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.last_layer_func(self.fc4(x))

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

#  Spijkervet / SimCLR / simclr / simclr.py
class SimCLR_net(nn.Module):

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR_net, self).__init__()

        self.encoder = encoder

        # We use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x_img_i, x_img_j, x_feat):

        h_i = self.encoder(x_img_i, x_feat)
        h_j = self.encoder(x_img_j, x_feat)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j

# -----------------------------------------------------------------------------

class Linear_classifier(nn.Module):

    def __init__(self, n_features, n_classes):
        super(Linear_classifier, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

# -----------------------------------------------------------------------------
