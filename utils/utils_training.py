import sys
import torch
import pickle

import pytorch_lightning as pl

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, ConfusionMatrix

# -----------------------------------------------------------------------------

sys.path.append('models')
from models import P_stamps_net, SimCLR_net, Linear_classifier

sys.path.append('losses')
from losses import P_stamps_loss, NT_Xent

sys.path.append('utils')
import utils_dataset
from utils_dataset import Dataset_stamps
from utils_plots import plot_confusion_matrix

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

class Supervised_Cross_Entropy(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        # Load params
        for key in args.keys():
            self.hparams[key] = args[key]

        # Initialize P-stamp network
        self.model = P_stamps_net(self.hparams.drop_ratio,
                                  n_features=5,
                                  last_act_function='Softmax')


    def forward(self, x_img, x_feat):

        y_pred  = self.model(x_img, x_feat)
        return y_pred


    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward
        x_img, x_feat ,y_true = batch
        y_pred = self.forward(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        return loss


    def prediction_target_loss(self, batch, batch_idx):

        x_img, x_feat ,y_true = batch
        y_pred = self.model(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss}


    def metrics(self, y_pred, y_true):

        metric_collection = MetricCollection([
            Accuracy(),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        metrics = metric_collection(y_pred, y_true)

        return metrics, conf_matrix(y_pred.cpu(), y_true.cpu())


    def validation_step(self, batch, batch_idx):

        output = self.prediction_target_loss(batch, batch_idx)

        return output


    def validation_epoch_end(self, validation_step_outputs):

        y_pred = torch.tensor([], dtype=torch.int8, device=self.hparams.device)
        y_true = torch.tensor([], dtype=torch.int8, device=self.hparams.device)
        loss = torch.tensor([],dtype=torch.float32, device=self.hparams.device)

        for output in validation_step_outputs:
            
            y_pred = torch.cat((y_pred, output['y_pred']), dim=0)
            y_true = torch.cat((y_true, output['y_true']), dim=0)

            loss = torch.cat((loss, torch.tensor([output['loss']], device=self.hparams.device)), dim=0)

        metrics, conf_mat = self.metrics(y_pred, y_true)
        metrics['loss'] = loss.mean()

        for key, value in metrics.items():
            self.log(key, value)

        acc = metrics['Accuracy']

        title = 'Average confusion matrix p-stamps (p-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/conf_mat_sce_Validation.png'

        plot_confusion_matrix(conf_mat, title, utils_dataset.label_names, file)

        return None


    def test_step(self, batch, batch_idx):

        output = self.prediction_target_loss(batch, batch_idx)

        return output


    def test_epoch_end(self, validation_step_outputs):


        y_pred = torch.tensor([], dtype=torch.int8, device=self.hparams.device)
        y_true = torch.tensor([], dtype=torch.int8, device=self.hparams.device)
        loss = torch.tensor([],dtype=torch.float32, device=self.hparams.device)

        for output in validation_step_outputs:
            
            y_pred = torch.cat((y_pred, output['y_pred']), dim=0)
            y_true = torch.cat((y_true, output['y_true']), dim=0)

            loss = torch.cat((loss, torch.tensor([output['loss']], device=self.hparams.device)), dim=0)

        metrics, conf_mat = self.metrics(y_pred, y_true)
        metrics['loss'] = loss.mean()

        for key, value in metrics.items():
            self.log(key, value)

        acc = metrics['Accuracy']

        title = 'Average confusion matrix p-stamps (p-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/conf_mat_sce_Test.png'
        plot_confusion_matrix(conf_mat, title, utils_dataset.label_names, file)

        return None


    def predict_step(self, batch, batch_idx, dataloader_idx):

        x_img, x_feat, y_true = batch
        print("predict_step:", self.model.training)
        return self.model(x_img, x_feat)


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.hparams.batch_size, self.hparams.beta)


    def prepare_data(self):

        transform_stamp = transforms.Compose([
            transforms.Resize(self.hparams.image_size),
            transforms.ToTensor()
            ])

        # Data reading
        self.training_data = Dataset_stamps(data,
                                            'Train',
                                            transform=transform_stamp,
                                            target_transform=None)

        # Data reading
        self.validation_data = Dataset_stamps(data,
                                              'Validation',
                                              transform=transform_stamp,
                                              target_transform=None)

        # Data reading
        self.test_data = Dataset_stamps(data,
                                        'Test',
                                        transform=transform_stamp,
                                        target_transform=None)


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(self.training_data,
                                      batch_size=self.hparams.batch_size,
                                      shuffle=True,
                                      num_workers=self.hparams.workers,
                                      pin_memory=False,
                                      drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True,
                                           num_workers=self.hparams.workers,
                                           drop_last=True)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=self.hparams.batch_size,
                                     shuffle=True,
                                     num_workers=self.hparams.workers,
                                     drop_last=True)

        return test_dataloader


    def configure_optimizers(self):

        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr,
                                          betas=(0.5, 0.9))

        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr)

        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=0.5)

        return optimizer

# -----------------------------------------------------------------------------

class Self_Supervised_CLR(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        # Load params
        for key in args.keys():
            self.hparams[key] = args[key]

        # Initialize P-stamp network
        encoder = P_stamps_net(self.hparams.drop_ratio,
                               self.hparams.n_features,
                               'Identity')

        # Initialize SimCLR network
        self.model = SimCLR_net(encoder,
                                self.hparams.projection_dim,
                                self.hparams.n_features)


    def setup(self, stage=None):

        self.criterion = NT_Xent(self.hparams.batch_size,
                                 self.hparams.temperature)


    def forward(self, x_img_i, x_img_j, x_feat):

        h_i, h_j, z_i, z_j = self.model(x_img_i, x_img_j, x_feat)
        return h_i, h_j, z_i, z_j


    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward
        (x_img_i, x_img_j), x_feat ,_ = batch
        h_i, h_j, z_i, z_j = self.forward(x_img_i, x_img_j, x_feat)
        loss = self.criterion(z_i, z_j)

        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx):

        x_img, x_feat, y_true = batch
        print("predict_step:", self.model.training)
        return self.model(x_img, x_feat)


    def configure_optimizers(self):

        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr,
                                          betas=(0.5, 0.9))

        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr)

        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                  lr=self.hparams.lr,
                                  momentum=0.5)

        return optimizer


# -----------------------------------------------------------------------------