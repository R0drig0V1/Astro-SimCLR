import torch
import pickle
import pl_bolts
import utils.dataset
import torchvision

import pytorch_lightning as pl

from losses import P_stamps_loss, NT_Xent, SupConLoss
from models import *

from torchlars import LARS

from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, ConfusionMatrix

from utils.args import Args
from utils.config import config
from utils.dataset import Dataset_stamps, BalancedBatchSampler
from utils.plots import plot_confusion_matrix
from utils.transformations import Augmentation_SimCLR, Resize_img

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------


class Supervised_Cross_Entropy(pl.LightningModule):

    def __init__(self, image_size, batch_size, drop_rate, beta_loss, lr,
                 optimizer, balanced_batch=False):
        super().__init__()
        
        # Load params
        self.image_size = image_size
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer

        # Initialize P-stamp network
        self.model = P_stamps_net(self.drop_rate,
                                  n_features=5,
                                  last_act_function='Softmax')

        self.balanced_batch = balanced_batch

        self.save_hyperparameters()


    def forward(self, x_img, x_feat):

        y_pred  = self.model(x_img, x_feat)
        return y_pred


    def training_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def training_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Train", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Train", metrics['Recall'], self.current_epoch)

        return None


    def validation_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def validation_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Validation", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Validation", metrics['Recall'], self.current_epoch)
        self.log('accuracy_val', metrics['Accuracy'], logger=False)

        return None


    def test_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def test_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Test", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Test", metrics['Recall'], self.current_epoch)

        return None


    def prediction_target_loss(self, batch, batch_idx):

        # Training_step defined the train loop
        x_img, x_feat, y_true = batch
        y_pred = self.forward(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss}


    def metrics(self, y_pred, y_true):

        # Inicializes metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Inicializes confusion matrix
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').to(y_pred.device)

        # Computes metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    def conf_mat_val(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.val_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():
                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true) * 100
        conf_mat = conf_matrix(y_pred, y_true)

        title = 'Confusion matrix P-stamps (P-stamps loss)\n Accuracy:{0:.2f}%'.format(acc)
        file = 'Figures/confusion_matrix_CE_Validation.png'
        plot_confusion_matrix(conf_mat, title, file)

        return None


    def conf_mat_test(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.test_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():
                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true) * 100
        conf_mat = conf_matrix(y_pred, y_true)

        title = 'Confusion matrix P-stamps (P-stamps loss)\n Accuracy:{0:.2f}%'.format(acc)
        file = 'Figures/confusion_matrix_CE_Test.png'
        plot_confusion_matrix(conf_mat, title, file)

        return None


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    def prepare_data(self):


        # Data reading
        self.training_data = Dataset_stamps(
                            data,
                            'Train',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        return None


    def train_dataloader(self):

        if self.balanced_batch:

            batch_sampler = BalancedBatchSampler(self.training_data,
                                                 n_classes=5,
                                                 n_samples=self.batch_size//5)

            train_dataloader = DataLoader(self.training_data,
                                          num_workers=config.workers,
                                          pin_memory=False,
                                          batch_sampler=batch_sampler)

        else:
            # Data loader
            train_dataloader = DataLoader(self.training_data,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=config.workers,
                                          pin_memory=False,
                                          drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data,
                                           batch_size=100,
                                           num_workers=config.workers)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=100,
                                     num_workers=config.workers)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)

        return optimizer

# -----------------------------------------------------------------------------

class Self_Supervised_SimCLR(pl.LightningModule):

    def __init__(self, image_size, batch_size, drop_rate, n_features,
                 projection_dim, temperature, lr, optimizer):
        super().__init__()
         
        # Load params
        self.image_size = image_size
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.n_features = n_features
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.lr = lr
        self.optimizer = optimizer

        # Initialize P-stamp network
        encoder = P_stamps_net(self.drop_rate,
                               self.n_features,
                               'Identity')

        # Initialize SimCLR network
        self.model = SimCLR(encoder,
                            self.projection_dim,
                            self.n_features)

        self.save_hyperparameters()


    def forward(self, x_img_i, x_img_j, x_feat):

        h_i, h_j, z_i, z_j = self.model(x_img_i, x_img_j, x_feat)
        return h_i, h_j, z_i, z_j


    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward
        (x_img_i, x_img_j), x_feat ,_ = batch
        h_i, h_j, z_i, z_j = self.forward(x_img_i, x_img_j, x_feat)
        loss = self.criterion(z_i, z_j)

        return loss


    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

        return None


    def validation_step(self, batch, batch_idx):

        (x_img_i, x_img_j), x_feat ,_ = batch
        h_i, h_j, z_i, z_j = self.forward(x_img_i, x_img_j, x_feat)
        loss = self.criterion(z_i, z_j)

        return loss


    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.log('loss_val', avg_loss, logger=False)

        return None


    def setup(self, stage=None):

        self.criterion = NT_Xent(self.batch_size,self.temperature)


    def prepare_data(self):

        # Data reading
        self.training_data_aug = Dataset_stamps(
                data,
                'Train',
                transform=Augmentation_SimCLR(size=self.image_size),
                one_hot_encoding=False)


        # Data reading
        self.validation_data_aug = Dataset_stamps(
                data,
                'Validation',
                transform=Augmentation_SimCLR(size=self.image_size),
                one_hot_encoding=False)


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(self.training_data_aug,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=config.workers,
                                      pin_memory=False,
                                      drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data_aug,
                                           batch_size=100,
                                           num_workers=config.workers)

        return validation_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)


        return optimizer


# -----------------------------------------------------------------------------

class Linear_SimCLR(pl.LightningModule):

    def __init__(self, simclr_model, image_size, batch_size, n_features,
                 beta_loss, lr, optimizer):
        super().__init__()
        

        # Load params
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer

        # SimCLR model is used
        self.encoder = simclr_model

        # Initialize Linear classifier
        self.model = Linear_classifier(n_features=self.n_features,
                                       n_classes=5)


        self.save_hyperparameters("image_size", "batch_size", "n_features",
                                  "beta_loss", "lr", "optimizer")


    def forward(self, x_img, x_feat):

        self.encoder.eval()

        with torch.no_grad():
            h, _, z, _ = self.encoder(x_img, x_img, x_feat)
        
        y_pred = self.model(h.detach())

        return y_pred


    def training_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def training_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Train", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Train", metrics['Recall'], self.current_epoch)

        return None


    def validation_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def validation_epoch_end(self, outputs):


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Validation", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Validation", metrics['Recall'], self.current_epoch)
        self.log('Accuracy', metrics['Accuracy'], logger=False)

        return None


    def test_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def test_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Test", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Test", metrics['Recall'], self.current_epoch)

        return None


    def prediction_target_loss(self, batch, batch_idx):

        # Training_step defined the train loop
        x_img, x_feat ,y_true = batch
        y_pred = self.forward(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss}


    def metrics(self, y_pred, y_true):

        # Inicializes metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Computes metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    def conf_mat_val(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.val_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():

                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true)
        conf_mat = conf_matrix(y_pred, y_true)

        title = 'Confusion matrix Self-Supervised \n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/confusion_matrix_SS_CLR_Validation.png'
        plot_confusion_matrix(conf_mat, title, utils.dataset.label_names, file)

        return None


    def conf_mat_test(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.test_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():

                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true)
        conf_mat = conf_matrix(y_pred, y_true)

        title = 'Confusion matrix Self-Supervised \n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/confusion_matrix_SS_CLR_Test.png'
        plot_confusion_matrix(conf_mat, title, utils.dataset.label_names, file)

        return None


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    def prepare_data(self):

        # Data reading
        self.training_data = Dataset_stamps(
                            data,
                            'Train',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(self.training_data,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=config.workers,
                                      pin_memory=False,
                                      drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data,
                                           batch_size=100,
                                           num_workers=config.workers)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=100,
                                     num_workers=config.workers)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)


        return optimizer


# -----------------------------------------------------------------------------

class CLR_a(pl.LightningModule):

    def __init__(self, encoder_name, image_size, batch_size,
                 projection_dim, temperature, lr, optimizer, method):


        super().__init__()
         
        # Load params
        self.encoder_name = encoder_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.lr = lr
        self.optimizer = optimizer
        self.method = method


        if (self.method=='supcon'):
            self.criterion = SupConLoss(self.temperature)
            #self.lr *= 100 if (self.encoder_name == 'resnet18') else 1000

        elif (self.method == 'simclr'):
            self.criterion = NT_Xent(self.batch_size, self.temperature)


        if (self.encoder_name == 'pstamps'):

            # Initialize P-stamp network (Part A)
            encoder = P_stamps_net_a()

            # Output features of the encoder
            self.n_features = encoder.fc1.out_features

            # Initialize SimCLR network
            self.model = CLR(encoder=encoder,
                             n_features=self.n_features,
                             projection_dim=self.projection_dim)


        elif (self.encoder_name == 'resnet18'):

            # Initialize resnet18
            encoder = torchvision.models.resnet18()

            # Output features of the encoder
            self.n_features = encoder.fc.in_features
            encoder.fc = Identity()

            # Initialize SimCLR network
            self.model = CLR(encoder=encoder,
                             n_features=self.n_features,
                             projection_dim=self.projection_dim)


        elif (self.encoder_name == 'resnet50'):

            # Initialize resnet18
            encoder = torchvision.models.resnet50()

            self.n_features = encoder.fc.in_features
            encoder.fc = Identity()

            # Initialize SimCLR network
            self.model = CLR(encoder=encoder,
                             n_features=self.n_features,
                             projection_dim=self.projection_dim)


        self.save_hyperparameters('encoder_name',
                                  'image_size',
                                  'batch_size',
                                  'projection_dim',
                                  'temperature',
                                  'lr',
                                  'optimizer',
                                  'method')


    def forward(self, x_img_i, x_img_j):

        h_i, h_j, z_i, z_j = self.model(x_img_i, x_img_j)
        return h_i, h_j, z_i, z_j


    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward
        (x_img_i, x_img_j), _, y_true = batch
        h_i, h_j, z_i, z_j = self.forward(x_img_i, x_img_j)

        loss = self.criterion(z_i, z_j, y_true)

        return loss


    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

        return None


    def validation_step(self, batch, batch_idx):

        (x_img_i, x_img_j), _, y_true = batch
        h_i, h_j, z_i, z_j = self.forward(x_img_i, x_img_j)

        loss = self.criterion(z_i, z_j, y_true)

        return loss


    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.log('loss_val', avg_loss, logger=False)

        return None


    def prepare_data(self):

        # Data reading
        self.training_data_aug = Dataset_stamps(
                data,
                'Train',
                transform=Augmentation_SimCLR(size=self.image_size),
                one_hot_encoding=False)


        # Data reading
        self.validation_data_aug = Dataset_stamps(
                data,
                'Validation',
                transform=Augmentation_SimCLR(size=self.image_size),
                one_hot_encoding=False)


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(self.training_data_aug,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=config.workers,
                                      pin_memory=False,
                                      drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data_aug,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=config.workers)

        return validation_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)


        return optimizer


# -----------------------------------------------------------------------------

class CLR_b(pl.LightningModule):

    def __init__(self, clr_model, image_size, batch_size, beta_loss, lr,
                 drop_rate, optimizer, with_features):
        super().__init__()
        

        # Hyperparameters of class are saved
        self.clr_model = clr_model
        self.image_size = image_size
        self.batch_size = batch_size
        self.beta_loss = beta_loss
        self.lr = lr
        self.drop_rate = drop_rate
        self.optimizer = optimizer
        self.with_features = with_features

        # Useful hyperparameters of encoder
        self.encoder_name = self.clr_model.encoder_name 
        self.encoder_loss = self.clr_model.method


        # dataset features are included to predict
        n_features_dataset = 26 if with_features else 0


        # Part B of P_stamps net
        if (self.encoder_name == 'pstamps'):

            n_features = self.clr_model.n_features + n_features_dataset
            self.model = P_stamps_net_b(drop_rate=self.drop_rate, n_features=n_features)

            
        # Initialize Linear classifier
        elif (self.encoder_name == 'resnet18'):

            n_features = self.clr_model.n_features + n_features_dataset
            self.model = Linear_classifier(n_features=n_features, n_classes=5)


        # Initialize Linear classifier for resnet50
        elif(self.encoder_name == 'resnet50'):

            n_features = self.clr_model.n_features  + n_features_dataset
            self.model = Linear_classifier(n_features=n_features, n_classes=5)


        self.save_hyperparameters(
            "image_size",
            "batch_size",
            "beta_loss",
            "lr",
            "drop_rate",
            "optimizer",
            "with_features"
        )


    def forward(self, x_img, x_feat):

        self.clr_model.eval()

        with torch.no_grad():
            h, _, z, _ = self.clr_model(x_img, x_img)
        

        # Features computed from image and features of dataset are concatenated
        if (self.with_features):
            x = torch.cat((h, x_feat), dim=1).detach()

        # Features of dataset are not used
        else:
            x = h.detach()

        y_pred = self.model(x)

        return y_pred


    def training_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def training_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Train", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Train", metrics['Recall'], self.current_epoch)

        return None


    def validation_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def validation_epoch_end(self, outputs):


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Validation", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Validation", metrics['Recall'], self.current_epoch)
        self.log('accuracy_val', metrics['Accuracy'], logger=False)

        return None


    def test_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    def test_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Test", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Test", metrics['Recall'], self.current_epoch)

        return None


    def prediction_target_loss(self, batch, batch_idx):

        # Training_step defined the train loop
        x_img, x_feat ,y_true = batch

        y_pred = self.forward(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss}


    def metrics(self, y_pred, y_true):

        # Inicializes metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Computes metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    def conf_mat_val(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.val_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():

                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true) * 100
        conf_mat = conf_matrix(y_pred, y_true)

        # Title of matrix
        title1 = "Confusion matrix CLR\n"
        title2 = f"Accuracy validation:{acc:.2f}%\n"
        features = 'with features' if self.with_features else 'without features'
        title3 = f"({self.encoder_name}, {features}, {self.encoder_loss})"
        title = title1 + title2 + title3

        # File name
        file = f'Figures/conf_mat_CLR_{self.encoder_name}_features-{self.with_features}_loss-{self.encoder_loss}_validation.png'

        # It plots matrix
        plot_confusion_matrix(conf_mat, title, file)
        return None


    def conf_mat_test(self):

        outputs = []

        self.model.eval()

        for idx, batch in enumerate(self.test_dataloader()):

            # Training_step defined the train loop
            x_img, x_feat, y_true = batch

            with torch.no_grad():

                y_pred = self.forward(x_img, x_feat)

            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)

        # Transforms to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()

        # Inicializes accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix = ConfusionMatrix(num_classes=5, normalize='true').cpu()

        acc = accuracy(y_pred, y_true) * 100
        conf_mat = conf_matrix(y_pred, y_true)
  
        # Title of matrix
        title1 = "Confusion matrix CLR\n"
        title2 = f"Accuracy test:{acc:.2f}%\n"
        features = 'with features' if self.with_features else 'without features'
        title3 = f"({self.encoder_name}, {features}, {self.encoder_loss})"
        title = title1 + title2 + title3

        # File name
        file = f'Figures/conf_mat_CLR_{self.encoder_name}_features-{self.with_features}_loss-{self.encoder_loss}_test.png'

        # It plots matrix
        plot_confusion_matrix(conf_mat, title, file)

        return None


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    def prepare_data(self):

        # Data reading
        self.training_data = Dataset_stamps(
                            data,
                            'Train',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            transform=Resize_img(size=self.image_size),
                            one_hot_encoding=True)


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(self.training_data,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=config.workers,
                                      pin_memory=False,
                                      drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        validation_dataloader = DataLoader(self.validation_data,
                                           batch_size=100,
                                           num_workers=config.workers)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=100,
                                     num_workers=config.workers)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)


        return optimizer

# -----------------------------------------------------------------------------

