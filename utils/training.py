import torch
import pickle
import utils.dataset

import pytorch_lightning as pl

from losses import P_stamps_loss, NT_Xent
from models import P_stamps_net, SimCLR_net, Linear_classifier
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, ConfusionMatrix
from utils.args import Args
from utils.dataset import Dataset_stamps
from utils.plots import plot_confusion_matrix
from utils.transformations import Augmentation_SimCLR, Resize_img

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

# Configurations
config = Args({'num_nodes': 1,
               'gpus': 1,
               'workers': 4,
               'model_path': "../weights"
               })

# -----------------------------------------------------------------------------


class Supervised_Cross_Entropy(pl.LightningModule):

    def __init__(self, image_size, batch_size, drop_rate, beta_loss, lr,
                 optimizer):
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

        acc = accuracy(y_pred, y_true)
        conf_mat = conf_matrix(y_pred, y_true)

        title = 'Confusion matrix P-stamps (P-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/confusion_matrix_S_CE_Validation.png'
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

        title = 'Confusion matrix P-stamps (P-stamps loss)\n Accuracy:{0:.2f}%'.format(acc*100)
        file = 'Figures/confusion_matrix_S_CE_Test.png'
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
                            target_transform=None)

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            transform=Resize_img(size=self.image_size),
                            target_transform=None)

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            transform=Resize_img(size=self.image_size),
                            target_transform=None)


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
                                           batch_size=self.batch_size,
                                           #shuffle=True,
                                           num_workers=config.workers,
                                           drop_last=True)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=self.batch_size,
                                     #shuffle=True,
                                     num_workers=config.workers,
                                     drop_last=True)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.9))

        elif self.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=0.5)

        return optimizer

# -----------------------------------------------------------------------------

class Self_Supervised_SimCLR(pl.LightningModule):

    def __init__(self,image_size, batch_size, drop_rate, n_features,
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
        self.model = SimCLR_net(encoder,
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
                target_transform=None)


        # Data reading
        self.validation_data_aug = Dataset_stamps(
                data,
                'Validation',
                transform=Augmentation_SimCLR(size=self.image_size),
                target_transform=None)


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
                                           batch_size=self.batch_size,
                                           #shuffle=True,
                                           num_workers=config.workers,
                                           drop_last=True)

        return validation_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.9))

        elif self.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=0.5)

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
        
        y_pred = self.model(h)

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
                            target_transform=None)

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            transform=Resize_img(size=self.image_size),
                            target_transform=None)

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            transform=Resize_img(size=self.image_size),
                            target_transform=None)


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
                                           batch_size=self.batch_size,
                                           #shuffle=True,
                                           num_workers=config.workers,
                                           drop_last=True)

        return validation_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=self.batch_size,
                                     #shuffle=True,
                                     num_workers=config.workers,
                                     drop_last=True)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.9))

        elif self.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=0.5)

        return optimizer

# -----------------------------------------------------------------------------