import torch
import pickle
import pl_bolts
import utils.dataset
import torchvision

import pytorch_lightning as pl
import torchvision.transforms as transforms

from losses import P_stamps_loss, NT_Xent, SupConLoss
from models import *

from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, ConfusionMatrix

from utils.config import config
from utils.dataset import Dataset_stamps, BalancedBatchSampler
from utils.plots import plot_confusion_matrix
from utils.transformations import Augmentation_SimCLR, Resize_img
from utils.transformations import Astro_Augmentation_SimCLR
from utils.transformations import Jitter_Astro_Aug
from utils.transformations import Jitter_Default_Aug
from utils.transformations import Crop_Astro_Aug
from utils.transformations import Crop_Default_Aug
from utils.transformations import Rotation_Aug

# -----------------------------------------------------------------------------

# Dataset is loaded
with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
    data = pickle.load(f)

# -----------------------------------------------------------------------------

class Supervised_Cross_Entropy(pl.LightningModule):

    def __init__(
            self,
            image_size,
            batch_size,
            drop_rate,
            beta_loss,
            lr,
            optimizer,
            with_features=True,
            balanced_batch=False,
            augmentation=False):

        super().__init__()
        
        # Load params
        self.image_size = image_size
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer
        self.with_features = with_features
        self.balanced_batch = balanced_batch
        self.augmentation = augmentation

        # Initialize P-stamp network
        self.model = P_stamps_net(self.drop_rate, self.with_features)

        # Save hyperparameters
        self.save_hyperparameters()


    def forward(self, x_img, x_feat):

        y_pred  = self.model(x_img, x_feat)
        return y_pred


    # Training loop
    def training_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    # End of training loop
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


    # Validation loop
    def validation_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx)


    # End of validation loop
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


    # Compute y_pred, y_true and loss for the batch
    def prediction_target_loss(self, batch, batch_idx):

        # Evaluate batch
        x_img, x_feat, y_true = batch
        y_pred = self.forward(x_img, x_feat)
        loss = self.criterion(y_pred, y_true)

        # Transform to scalar labels
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss}


    # Compute accuracy, precision and recall
    def metrics(self, y_pred, y_true):

        # Inicialize metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Compute metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    # Compute confusion matrix and accuracy
    def confusion_matrix(self, dataset):

        # Load dataset
        if (dataset=='Train'):
            dataloader = self.train_dataloader()

        elif (dataset=='Validation'):
            dataloader = self.val_dataloader()

        elif (dataset=='Test'):
            dataloader = self.test_dataloader()


        # evaluation mode
        self.model.eval()


        # save y_true and y_pred of dataloader
        outputs = []


        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                y_pred = self.forward(x_img, x_feat)

            # Save output
            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        # Concatenate results
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)


        # Transform output to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()


        # Inicialize accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix_func = ConfusionMatrix(num_classes=5, normalize='true').cpu()


        # Compute accuracy and confusion matrix
        acc = accuracy(y_pred, y_true).numpy()
        conf_mat = conf_matrix_func(y_pred, y_true).numpy()

        return acc, conf_mat


    # Plot confusion matrix
    def plot_confusion_matrix(self, dataset):

        # Compute accuracy and confusion matrix
        acc, conf_mat = self.confusion_matrix(dataset=dataset)

        # Plot confusion matrix for dataset
        title = f'Confusion matrix P-stamps (P-stamps loss)\n Accuracy {dataset}:{acc:.3f}'
        file = f'Figures/confusion_matrix_CE_{dataset}.png'
        plot_confusion_matrix(conf_mat, title, file)

        return None


    # Inicialize loss function
    def setup(self, stage=None):
        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    # Prepare datasets
    def prepare_data(self):

        if self.augmentation == 'astro':
            augmentation = Augmentation_SimCLR(size=self.image_size)

        elif self.augmentation == 'default':
            augmentation = Astro_Augmentation_SimCLR(size=self.image_size)

        elif self.augmentation == False:
            augmentation = None


        # Data reading
        self.training_data = Dataset_stamps(
                            data,
                            'Train',
                            image_size=self.image_size,
                            image_transformation=augmentation,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps(
                            data,
                            'Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps(
                            data,
                            'Test',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        return None


    # Dataloader of train dataset
    def train_dataloader(self):

        # dataloader with same samples of each class
        if self.balanced_batch:

            batch_sampler = BalancedBatchSampler(self.training_data,
                                                 n_classes=5,
                                                 n_samples=self.batch_size//5)

            train_dataloader = DataLoader(self.training_data,
                                          num_workers=config.workers,
                                          pin_memory=False,
                                          batch_sampler=batch_sampler)

        else:
            
            train_dataloader = DataLoader(self.training_data,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=config.workers,
                                          pin_memory=False,
                                          drop_last=True)

        return train_dataloader


    # Dataloader of validation dataset
    def val_dataloader(self):

        val_dataloader = DataLoader(self.validation_data,
                                    batch_size=100,
                                    num_workers=config.workers)

        return val_dataloader


    # Dataloader of test dataset
    def test_dataloader(self):

        test_dataloader = DataLoader(self.test_data,
                                     batch_size=100,
                                     num_workers=config.workers)

        return test_dataloader


    # Configuration of optimizer
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

class SimCLR_classifier(pl.LightningModule):

    def __init__(self, simclr_model, with_features):
        super().__init__()

        # Hyperparameters of class are saved
        self.simclr_model = simclr_model
        self.image_size = self.simclr_model.image_size
        self.batch_size = self.simclr_model.batch_size_classifier
        self.beta_loss = self.simclr_model.beta_loss
        self.lr = self.simclr_model.lr_classifier
        self.optimizer = self.simclr_model.optimizer_classifier
        self.with_features = with_features

        # dataset features are included to predict
        n_features_dataset = 23 if with_features else 0

        # Initialize classifier
        n_features = self.simclr_model.n_features_encoder + n_features_dataset
        self.classifier = Linear_classifier(input_size=n_features, n_classes=5)

        self.save_hyperparameters(
            "with_features"
        )


    def forward(self, x_img, x_feat):

        self.simclr_model.eval()

        with torch.no_grad():
            h, _, z, _ = self.simclr_model.CLR(x_img, x_img)
        
        # Features computed from image and features of dataset are concatenated
        if (self.with_features): x = torch.cat((h, x_feat), dim=1).detach()

        # Features of dataset are not used
        else: x = h.detach()

        y_pred = self.classifier(x)

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

        # Inicialize metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Computes metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    def confusion_matrix(self, dataset):

        # Load dataset
        if (dataset=='Train'):
            dataloader = self.train_dataloader()

        elif (dataset=='Validation'):
            dataloader = self.val_dataloader()

        elif (dataset=='Test'):
            dataloader = self.test_dataloader()


        # Evaluation mode
        self.simclr_model.CLR.eval()
        self.classifier.eval()


        # save y_true and y_pred of dataloader
        outputs = []


        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                y_pred = self.forward(x_img, x_feat)

            # Save output
            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        # Concatenate results
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)


        # Transform output to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()


        # Inicialize accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix_func = ConfusionMatrix(num_classes=5, normalize='true').cpu()


        # Compute accuracy and confusion matrix
        acc = accuracy(y_pred, y_true).numpy()
        conf_mat = conf_matrix_func(y_pred, y_true).numpy()

        return acc, conf_mat


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    def prepare_data(self):

        # Data reading
        self.training_data = Dataset_stamps(
                            pickle=data,
                            dataset='Train',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps(
                            pickle=data,
                            dataset='Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps(
                            pickle=data,
                            dataset='Test',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=False,
            drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        val_dataloader = DataLoader(
            self.validation_data,
            batch_size=100,
            num_workers=config.workers)

        return val_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(
            self.test_data,
            batch_size=100,
            num_workers=config.workers)

        return test_dataloader


    def configure_optimizers(self):

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.classifier.parameters(), lr=self.lr)

        return optimizer


# -----------------------------------------------------------------------------

class SimCLR(pl.LightningModule):

    def __init__(
            self,
            encoder_name,
            method,
            image_size,
            astro_augmentation,
            projection_dim,
            temperature,
            lr_encoder,
            batch_size_encoder,
            optimizer_encoder,
            beta_loss,
            lr_classifier,
            batch_size_classifier,
            optimizer_classifier,
            with_features=True):

        super().__init__()
        

        # Hyperparameters of class are saved
        self.encoder_name = encoder_name
        self.method = method
        self.image_size = image_size
        self.astro_augmentation = astro_augmentation
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.lr_encoder = lr_encoder
        self.batch_size_encoder = batch_size_encoder
        self.optimizer_encoder = optimizer_encoder
        self.beta_loss = beta_loss
        self.lr_classifier = lr_classifier
        self.batch_size_classifier = batch_size_classifier
        self.optimizer_classifier = optimizer_classifier
        self.with_features = with_features
        self.batch_size = batch_size_encoder

        # Encoder loss
        if (self.method=='supcon'):
            self.criterion_encoder = SupConLoss(self.temperature)

        elif (self.method == 'simclr'):
            self.criterion_encoder = NT_Xent(self.batch_size_encoder, self.temperature)


        # Classifier loss
        self.criterion_classifier = P_stamps_loss(self.batch_size_classifier, self.beta_loss)


        # Dataset features are included to predict
        n_features_dataset = 23 if with_features else 0


        if (self.encoder_name == 'pstamps'):

            # Initialize P-stamp network
            self.encoder = P_stamps_net_SimCLR()

            # Output features of the encoder
            self.n_features_encoder = self.encoder.fc3.out_features


        elif (self.encoder_name == 'resnet18'):

            # Initialize resnet18
            self.encoder = torchvision.models.resnet18()

            # Output features of the encoder
            self.n_features_encoder = self.encoder.fc.in_features

            # Last layer of the encoder is removed
            self.encoder.fc = Identity()


        elif (self.encoder_name == 'resnet50'):

            # Initialize resnet50
            self.encoder = torchvision.models.resnet50()

            # Output features of the encoder
            self.n_features_encoder = self.encoder.fc.in_features

            # Last layer of the encoder is removed
            self.encoder.fc = Identity()


        # Initialize SimCLR network
        self.CLR = CLR(
            encoder=self.encoder,
            input_size=self.n_features_encoder,
            projection_size=self.projection_dim)


        # Initialize classifier
        input_size_classifier = self.n_features_encoder + n_features_dataset
        self.classifier = Linear_classifier(
            input_size=input_size_classifier,
            n_classes=5)


        # Save hyperparameters
        self.save_hyperparameters()


    def forward(self, x_img, x_img_i, x_img_j, x_feat):

        # Forward of SimCLR 
        h_i, h_j, z_i, z_j = self.CLR(x_img_i, x_img_j)

        # CLR is set to evaluation mode
        self.CLR.eval()

        with torch.no_grad():
            h, _, z, _ = self.CLR(x_img, x_img)

        # Features computed from image and features of dataset are concatenated
        if self.with_features: x = torch.cat((h, x_feat), dim=1).detach()

        # Features of dataset are not used
        else: x = h.detach()

        # Prediction
        y_pred = self.classifier(x)

        return (z_i, z_j), y_pred


    # Training loop
    def training_step(self, batch, batch_idx, optimizer_idx):

        self.CLR.train()
        return self.prediction_target_loss(batch, batch_idx, optimizer_idx)


    # End of training loop
    def training_epoch_end(self, outputs):

        # Some issue returns the output inside a list
        outputs = outputs[0]

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss_sum'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Train", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Train", metrics['Recall'], self.current_epoch)

        return None


    # Validation loop
    def validation_step(self, batch, batch_idx):

        return self.prediction_target_loss(batch, batch_idx, optimizer_idx=1)


    # End of validation loop
    def validation_epoch_end(self, outputs):

        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['loss_sum'] for x in outputs]).mean()

        metrics = self.metrics(y_pred, y_true)

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", metrics['Accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Validation", metrics['Precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Validation", metrics['Recall'], self.current_epoch)
        self.log('accuracy_val', metrics['Accuracy'], logger=False)

        return None


    # Compute y_pred, y_true and loss for the batch
    def prediction_target_loss(self, batch, batch_idx, optimizer_idx):

        # Evaluate batch
        x_img, (x_img_i, x_img_j), x_feat, y_true = batch
        (z_i, z_j), y_pred = self.forward(x_img, x_img_i, x_img_j, x_feat)

        # Compute loss of encoder
        loss_encoder = self.criterion_encoder(z_i, z_j, y_true)

        # Conpute loss of classifier
        y_pred_sub = y_pred[:,:self.batch_size_classifier]
        y_true_sub = torch.nn.functional.one_hot(y_true, num_classes=5)[:,:self.batch_size_classifier]
        loss_classifier = self.criterion_classifier(y_pred_sub, y_true_sub)

        # Total loss
        loss = loss_encoder + loss_classifier

        # Transform to scalar labels
        y_pred = torch.argmax(y_pred, dim=1)

        if optimizer_idx==0:
            return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss_encoder, 'loss_sum':loss.detach()}

        elif optimizer_idx==1:
            return {'y_pred': y_pred, 'y_true': y_true, 'loss': loss_classifier, 'loss_sum':loss.detach()}


    # Compute accuracy, precision and recall
    def metrics(self, y_pred, y_true):

        # Inicialize metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Compute metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    # Compute confusion matrix and accuracy
    def confusion_matrix(self, dataset):

        # Load dataset
        if (dataset=='Train'):
            dataloader = self.train_dataloader()

        elif (dataset=='Validation'):
            dataloader = self.val_dataloader()

        elif (dataset=='Test'):
            dataloader = self.test_dataloader()


        # evaluation mode
        self.CLR.eval()
        self.classifier.eval()


        # save y_true and y_pred of dataloader
        outputs = []


        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, (x_img_i, x_img_j), x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                (z_i, z_j), y_pred = self.forward(x_img, x_img_i, x_img_j, x_feat)

            # Save output
            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        # Concatenate results
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)


        # Transform output to scalar labels
        #y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()


        # Inicialize accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix_func = ConfusionMatrix(num_classes=5, normalize='true').cpu()


        # Compute accuracy and confusion matrix
        acc = accuracy(y_pred, y_true).numpy()
        conf_mat = conf_matrix_func(y_pred, y_true).numpy()

        return acc, conf_mat


    # Plot confusion matrix
    def plot_confusion_matrix(self, dataset):

        # Compute accuracy and confusion matrix
        acc, conf_mat = self.confusion_matrix(dataset=dataset)

        # Plot confusion matrix for dataset
        title = f'Confusion matrix SimCLR\n Accuracy {dataset}:{acc:.3f}'
        file = f'Figures/confusion_matrix_SimCLR_{dataset}.png'
        plot_confusion_matrix(conf_mat, title, file)

        return None


    # Prepare datasets
    def prepare_data(self):

        if self.astro_augmentation==True:
            augmentation = Augmentation_SimCLR(size=self.image_size)

        elif self.astro_augmentation==False:
            augmentation = Astro_Augmentation_SimCLR(size=self.image_size)

        elif self.astro_augmentation=='Jitter_astro':
            augmentation = Jitter_Astro_Aug(size=self.image_size)

        elif self.astro_augmentation=='Jitter_default':
            augmentation = Jitter_Default_Aug(size=self.image_size)

        elif self.astro_augmentation=='Crop_astro':
            augmentation = Crop_Astro_Aug(size=self.image_size)

        elif self.astro_augmentation=='Crop_default':
            augmentation = Crop_Default_Aug(size=self.image_size)
            
        elif self.astro_augmentation=='Rotation':
            augmentation = Rotation_Aug(size=self.image_size)

        # Data reading
        self.training_data_aug = Dataset_stamps(
            pickle=data,
            dataset='Train',
            image_size=self.image_size,
            image_transformation=augmentation,
            image_original_and_augmentated=True,
            one_hot_encoding=False,
            discarted_features=[13,14,15])


        # Data reading
        self.validation_data_aug = Dataset_stamps(
            pickle=data,
            dataset='Validation',
            image_size=self.image_size,
            image_transformation=augmentation,
            image_original_and_augmentated=True,
            one_hot_encoding=False,
            discarted_features=[13,14,15])


        # Data reading
        self.test_data_aug = Dataset_stamps(
            pickle=data,
            dataset='Test',
            image_size=self.image_size,
            image_transformation=augmentation,
            image_original_and_augmentated=True,
            one_hot_encoding=False,
            discarted_features=[13,14,15])


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(
            self.training_data_aug,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=False,
            drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        val_dataloader = DataLoader(
            self.validation_data_aug,
            batch_size=100,
            num_workers=config.workers)

        return val_dataloader


    # Dataloader of test dataset
    def test_dataloader(self):

        test_dataloader = DataLoader(
            self.test_data_aug,
            batch_size=100,
            num_workers=config.workers)

        return test_dataloader


    # Configuration of optimizer
    def configure_optimizers(self):

        if self.optimizer_encoder == "AdamW":
            optimizer_encoder = torch.optim.AdamW(self.CLR.parameters(), lr=self.lr_encoder)

        elif self.optimizer_encoder == "Adam":
            optimizer_encoder = torch.optim.Adam(self.CLR.parameters(), lr=self.lr_encoder)

        elif self.optimizer_encoder == "SGD":
            optimizer_encoder = torch.optim.SGD(self.CLR.parameters(), lr=self.lr_encoder)

        elif self.optimizer_encoder == "LARS":            
            optimizer_encoder = pl_bolts.optimizers.LARS(self.CLR.parameters(), lr=self.lr_encoder)


        if self.optimizer_classifier == "AdamW":
            optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr_classifier)

        elif self.optimizer_classifier == "Adam":
            optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=self.lr_classifier)

        elif self.optimizer_classifier == "SGD":
            optimizer_classifier = torch.optim.SGD(self.classifier.parameters(), lr=self.lr_classifier)

        elif self.optimizer_classifier == "LARS":            
            optimizer_classifier = pl_bolts.optimizers.LARS(self.classifier.parameters(), lr=self.lr_classifier)


        return [optimizer_encoder, optimizer_classifier]

# -----------------------------------------------------------------------------

class Fine_SimCLR(pl.LightningModule):

    def __init__(self, simclr_model, lr, batch_size, with_features):
        super().__init__()


        # Hyperparameters of class are saved
        self.image_size = simclr_model.image_size
        self.optimizer = simclr_model.optimizer_classifier
        self.beta_loss = simclr_model.beta_loss
        self.lr = lr
        self.batch_size = batch_size
        self.with_features = with_features


        # dataset features are included to predict
        n_features_dataset = 23 if with_features else 0

        # Number of features (linear classifier)
        n_features = simclr_model.n_features_encoder + n_features_dataset


        # Model
        self.model = torch.nn.Sequential(
            simclr_model.CLR.encoder,
            Linear_classifier(input_size=n_features, n_classes=5)
        )


        self.save_hyperparameters(
            "lr",
            "batch_size",
            "with_features"
        )


    def forward(self, x_img, x_feat):

        # The encoder is used to compute the features
        h = self.model[0](x_img)

        # Features computed from image and features of dataset are concatenated
        if (self.with_features): x = torch.cat((h, x_feat), dim=1)

        # Features of dataset are not used
        else: x = h

        y_pred = self.model[1](x)

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

        # Inicialize metrics
        metric_collection = MetricCollection([
            Accuracy(num_classes=5),
            Precision(num_classes=5, average='macro'),
            Recall(num_classes=5, average='macro')
            ]).to(y_pred.device)

        # Computes metrics
        metrics = metric_collection(y_pred, y_true)

        return metrics


    def confusion_matrix(self, dataset):

        # Load dataset
        if (dataset=='Train'):
            dataloader = self.train_dataloader()

        elif (dataset=='Validation'):
            dataloader = self.val_dataloader()

        elif (dataset=='Test'):
            dataloader = self.test_dataloader()


        # Evaluation mode
        self.model.eval()


        # save y_true and y_pred of dataloader
        outputs = []


        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                y_pred = self.forward(x_img, x_feat)

            # Save output
            outputs.append({'y_true':y_true, 'y_pred':y_pred})


        # Concatenate results
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)


        # Transform output to scalar labels
        y_true = torch.argmax(y_true, dim=1).cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()


        # Inicialize accuraccy and confusion matrix
        accuracy = Accuracy(num_classes=5).cpu()
        conf_matrix_func = ConfusionMatrix(num_classes=5, normalize='true').cpu()


        # Compute accuracy and confusion matrix
        acc = accuracy(y_pred, y_true).numpy()
        conf_mat = conf_matrix_func(y_pred, y_true).numpy()

        return acc, conf_mat


    def setup(self, stage=None):

        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    def prepare_data(self):

        # Data reading
        self.training_data = Dataset_stamps(
                            pickle=data,
                            dataset='Train',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps(
                            pickle=data,
                            dataset='Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps(
                            pickle=data,
                            dataset='Test',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])


    def train_dataloader(self):

        # Data loader
        train_dataloader = DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=False,
            drop_last=True)

        return train_dataloader


    def val_dataloader(self):

        # Data loader
        val_dataloader = DataLoader(
            self.validation_data,
            batch_size=100,
            num_workers=config.workers)

        return val_dataloader


    def test_dataloader(self):

        # Data loader
        test_dataloader = DataLoader(
            self.test_data,
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
