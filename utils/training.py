import torch
import pickle


import utils.dataset
import torchvision

import numpy as np
import pytorch_lightning as pl
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from losses import P_stamps_loss, NT_Xent#, SupConLoss
from models import *

from timm.scheduler import CosineLRScheduler #TanhLRScheduler, 
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, ConfusionMatrix

from sklearn.manifold import TSNE
#from umap import UMAP

from utils.config import config
from utils.dataset import Dataset_stamps, Dataset_stamps_v2, Dataset_simclr, BalancedBatchSampler, Batch_sampler_step
from utils.lars import LARS
from utils.plots import plot_confusion_matrix

from utils.transformations import SimCLR_augmentation
from utils.transformations import SimCLR_augmentation_v2
from utils.transformations import SimCLR_augmentation_v3
from utils.transformations import Astro_augmentation
from utils.transformations import Astro_augmentation_v0
from utils.transformations import Astro_augmentation_v2
from utils.transformations import Astro_augmentation_v3
from utils.transformations import Astro_augmentation_v4
from utils.transformations import Astro_augmentation_v5
from utils.transformations import Astro_augmentation_v6
from utils.transformations import Astro_augmentation_v7
from utils.transformations import Astro_augmentation_v8
from utils.transformations import Astro_augmentation_v9
from utils.transformations import Jitter_astro
from utils.transformations import Jitter_astro_v2
from utils.transformations import Jitter_astro_v3
from utils.transformations import Jitter_simclr
from utils.transformations import RandomGrayscale
from utils.transformations import Crop_astro
from utils.transformations import Crop_simclr
from utils.transformations import Rotation
from utils.transformations import Rotation_v2
from utils.transformations import Rotation_v3
from utils.transformations import Center_crop_rotation
from utils.transformations import Gaussian_blur
from utils.transformations import RandomPerspective
from utils.transformations import RotationPerspective
from utils.transformations import RotationPerspectiveBlur
from utils.transformations import GridDistortion
from utils.transformations import RotationGrid
from utils.transformations import RotationGridBlur
from utils.transformations import ElasticTransform
from utils.transformations import RotationElastic
from utils.transformations import RotationElasticBlur
from utils.transformations import ElasticGrid
from utils.transformations import ElasticPerspective
from utils.transformations import GridPerspective
from utils.transformations import RotElasticGridPerspective
from utils.transformations import Resize_img

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
            augmentation='without_aug',
            data_path='dataset/td_ztf_stamp_17_06_20.pkl'):

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
        self.data_path = data_path

        # Initialize P-stamp network
        self.model = P_stamps_net(self.drop_rate, self.with_features)

        # Save hyperparameters
        self.save_hyperparameters()


    def forward(self, x_img, x_feat):

        logits  = self.model(x_img, x_feat)

        return logits


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
        self.log('accuracy_val', metrics['Accuracy'], logger=False, prog_bar=True)

        return None


    # Compute y_pred, y_true and loss for the batch
    def prediction_target_loss(self, batch, batch_idx):

        # Evaluate batch
        x_img, x_feat, y_true = batch

        logits = self.forward(x_img, x_feat)
        loss = self.criterion(logits, y_true)
        y_pred = logits.softmax(dim=1)

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
                logits = self.forward(x_img, x_feat)
                y_pred = logits.softmax(dim=1)

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

        if self.augmentation == 'simclr':
            augmentation = SimCLR_augmentation(size=self.image_size)

        elif self.augmentation == 'astro':
            augmentation = Astro_augmentation(size=self.image_size)

        elif self.augmentation == 'jitter_astro':
            augmentation = Jitter_astro(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v2':
            augmentation = Jitter_astro_v2(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v3':
            augmentation = Jitter_astro_v3(size=self.image_size)

        elif self.augmentation == 'jitter_simclr':
            augmentation = Jitter_simclr(size=self.image_size)

        elif self.augmentation == 'gray_scale':
            augmentation = RandomGrayscale(size=self.image_size)

        elif self.augmentation == 'crop_astro':
            augmentation = Crop_astro(size=self.image_size)

        elif self.augmentation == 'crop_simclr':
            augmentation = Crop_simclr(size=self.image_size)
            
        elif self.augmentation == 'rotation':
            augmentation = Rotation(size=self.image_size)

        elif self.augmentation == 'rotation_v2':
            augmentation = Rotation_v2(size=self.image_size)

        elif self.augmentation == 'rotation_v3':
            augmentation = Rotation_v3(size=self.image_size)

        elif self.augmentation == 'crop_rotation':
            augmentation = Center_crop_rotation(size=self.image_size)

        elif self.augmentation == 'blur':
            augmentation = Gaussian_blur(size=self.image_size)

        elif self.augmentation == 'perspective':
            augmentation = RandomPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective':
            augmentation = RotationPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective_blur':
            augmentation = RotationPerspectiveBlur(size=self.image_size)

        elif self.augmentation == 'grid_distortion':
            augmentation = GridDistortion(size=self.image_size)

        elif self.augmentation == 'rot_grid':
            augmentation = RotationGrid(size=self.image_size)

        elif self.augmentation == 'rot_grid_blur':
            augmentation = RotationGridBlur(size=self.image_size)

        elif self.augmentation == 'elastic_transform':
            augmentation = ElasticTransform(size=self.image_size)

        elif self.augmentation == 'rot_elastic':
            augmentation = RotationElastic(size=self.image_size)

        elif self.augmentation == 'rot_elastic_blur':
            augmentation = RotationElasticBlur(size=self.image_size)

        elif self.augmentation == 'elastic_grid':
            augmentation = ElasticGrid(size=self.image_size)

        elif self.augmentation == 'elastic_prespective':
            augmentation = ElasticPerspective(size=self.image_size)

        elif self.augmentation == 'grid_perspective':
            augmentation = GridPerspective(size=self.image_size)

        elif self.augmentation == 'rot_elastic_grid_perspective':
            augmentation = RotElasticGridPerspective(size=self.image_size)

        elif self.augmentation == 'without_aug':
            augmentation = None


        # Data reading
        self.training_data = Dataset_stamps_v2(
                            self.data_path,
                            'Train',
                            image_size=self.image_size,
                            image_transformation=augmentation,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps_v2(
                            self.data_path,
                            'Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps_v2(
                            self.data_path,
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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)

        return optimizer

# -----------------------------------------------------------------------------

class Supervised_Cross_Entropy_Resnet(pl.LightningModule):

    def __init__(
            self,
            image_size,
            batch_size,
            beta_loss,
            lr,
            optimizer,
            with_features=True,
            balanced_batch=False,
            augmentation='without_aug',
            data_path='dataset/td_ztf_stamp_17_06_20.pkl',
            resnet_model='resnet18',
            drop_rate=None):

        super().__init__()
        
        # Load params
        self.image_size = image_size
        self.batch_size = batch_size
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer
        self.with_features = with_features
        self.balanced_batch = balanced_batch
        self.augmentation = augmentation
        self.data_path = data_path


        if resnet_model=='resnet18':
            model = torchvision.models.resnet18()

        elif resnet_model=='resnet50':
            model = torchvision.models.resnet50()

        elif resnet_model=='resnet152':
            model = torchvision.models.resnet152()


        # dataset features are included to predict
        n_features_dataset = 23 if with_features else 0

        # Number of features (linear classifier)
        n_features = model.fc.in_features + n_features_dataset

        model.fc = Identity()

        self.feature_extractor = model
        self.linear_classifier = Linear_classifier(
            input_size=n_features,
            n_classes=5)


        # Save hyperparameters
        self.save_hyperparameters()


    def forward(self, x_img, x_feat):

        x_img  = self.feature_extractor(x_img)

        if self.with_features:
            return self.linear_classifier(torch.cat((x_img, x_feat), dim=1))

        else:
            return self.linear_classifier(x_img)


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

        logits = self.forward(x_img, x_feat)
        loss = self.criterion(logits, y_true)
        y_pred = logits.softmax(dim=1)

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
        self.feature_extractor.eval()
        self.linear_classifier.eval()


        # save y_true and y_pred of dataloader
        outputs = []


        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                logits = self.forward(x_img, x_feat)
                y_pred = logits.softmax(dim=1)

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
        title = f'Confusion matrix Resnet18 (P-stamps loss)\n Accuracy {dataset}:{acc:.3f}'
        file = f'Figures/confusion_matrix_CE_Resnet18_{dataset}.png'
        plot_confusion_matrix(conf_mat, title, file)

        return None


    # Inicialize loss function
    def setup(self, stage=None):
        self.criterion = P_stamps_loss(self.batch_size, self.beta_loss)


    # Prepare datasets
    def prepare_data(self):

        if self.augmentation == 'simclr':
            augmentation = SimCLR_augmentation(size=self.image_size)

        elif self.augmentation == 'astro':
            augmentation = Astro_augmentation(size=self.image_size)

        elif self.augmentation == 'jitter_astro':
            augmentation = Jitter_astro(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v2':
            augmentation = Jitter_astro_v2(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v3':
            augmentation = Jitter_astro_v3(size=self.image_size)

        elif self.augmentation == 'jitter_simclr':
            augmentation = Jitter_simclr(size=self.image_size)

        elif self.augmentation == 'gray_scale':
            augmentation = RandomGrayscale(size=self.image_size)

        elif self.augmentation == 'crop_astro':
            augmentation = Crop_astro(size=self.image_size)

        elif self.augmentation == 'crop_simclr':
            augmentation = Crop_simclr(size=self.image_size)
            
        elif self.augmentation == 'rotation':
            augmentation = Rotation(size=self.image_size)

        elif self.augmentation == 'rotation_v2':
            augmentation = Rotation_v2(size=self.image_size)

        elif self.augmentation == 'rotation_v3':
            augmentation = Rotation_v3(size=self.image_size)

        elif self.augmentation == 'crop_rotation':
            augmentation = Center_crop_rotation(size=self.image_size)

        elif self.augmentation == 'blur':
            augmentation = Gaussian_blur(size=self.image_size)

        elif self.augmentation == 'perspective':
            augmentation = RandomPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective':
            augmentation = RotationPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective_blur':
            augmentation = RotationPerspectiveBlur(size=self.image_size)

        elif self.augmentation == 'grid_distortion':
            augmentation = GridDistortion(size=self.image_size)

        elif self.augmentation == 'rot_grid':
            augmentation = RotationGrid(size=self.image_size)

        elif self.augmentation == 'rot_grid_blur':
            augmentation = RotationGridBlur(size=self.image_size)

        elif self.augmentation == 'elastic_transform':
            augmentation = ElasticTransform(size=self.image_size)

        elif self.augmentation == 'rot_elastic':
            augmentation = RotationElastic(size=self.image_size)

        elif self.augmentation == 'rot_elastic_blur':
            augmentation = RotationElasticBlur(size=self.image_size)

        elif self.augmentation == 'elastic_grid':
            augmentation = ElasticGrid(size=self.image_size)

        elif self.augmentation == 'elastic_prespective':
            augmentation = ElasticPerspective(size=self.image_size)

        elif self.augmentation == 'grid_perspective':
            augmentation = GridPerspective(size=self.image_size)

        elif self.augmentation == 'rot_elastic_grid_perspective':
            augmentation = RotElasticGridPerspective(size=self.image_size)

        elif self.augmentation == 'without_aug':
            augmentation = None


        # Data reading
        self.training_data = Dataset_stamps_v2(
                            self.data_path,
                            'Train',
                            image_size=self.image_size,
                            image_transformation=augmentation,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps_v2(
                            self.data_path,
                            'Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps_v2(
                            self.data_path,
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
            optimizer = torch.optim.AdamW(list(self.feature_extractor.parameters())+ list(self.linear_classifier.parameters()), lr=self.lr)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(list(self.feature_extractor.parameters())+ list(self.linear_classifier.parameters()), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(list(self.feature_extractor.parameters())+ list(self.linear_classifier.parameters()), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(list(self.feature_extractor.parameters())+ list(self.linear_classifier.parameters()), lr=self.lr)

        return optimizer


# -----------------------------------------------------------------------------


class SimCLR_classifier(pl.LightningModule):

    def __init__(
            self,
            simclr_model,
            batch_size,
            drop_rate,
            beta_loss,
            lr,
            optimizer,
            with_features,
            augmentation,
            data_path='dataset/td_ztf_stamp_17_06_20.pkl'):

        super().__init__()

        # Hyperparameters of class are saved
        self.simclr_model = simclr_model
        self.image_size = self.simclr_model.image_size
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer
        self.with_features = with_features
        self.augmentation = augmentation
        self.data_path = data_path

        # dataset features are included to predict
        n_features_dataset = 23 if with_features else 0

        # Initialize classifier
        n_features = self.simclr_model.n_features_encoder + n_features_dataset
        self.classifier = Linear_classifier(input_size=n_features, n_classes=5)

        self.save_hyperparameters(
            "batch_size",
            "drop_rate",
            "beta_loss",
            "lr",
            "optimizer",
            "with_features",
            "augmentation",
            "data_path"
        )


    def forward(self, x_img, x_feat):

        self.simclr_model.eval()

        with torch.no_grad():
            h = self.simclr_model.encoder(x_img)
            h = h * (1/torch.norm(h, dim=1, keepdim=True))
        
        # Features computed from image and features of dataset are concatenated
        if (self.with_features): x = torch.cat((h, x_feat), dim=1).detach()

        # Features of dataset are not used
        else: x = h.detach()

        logits = self.classifier(x)

        return logits


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
        self.log('accuracy_val', metrics['Accuracy'], logger=False, prog_bar=True)
        self.log('loss_val', avg_loss, logger=False, prog_bar=True)

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

        logits = self.forward(x_img, x_feat)
        y_pred = logits.softmax(dim=1)
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
                logits = self.forward(x_img, x_feat)
                y_pred = logits.softmax(dim=1)

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


   # Prepare datasets
    def prepare_data(self):

        if self.augmentation == 'simclr':
            augmentation = SimCLR_augmentation(size=self.image_size)

        elif self.augmentation == 'astro':
            augmentation = Astro_augmentation(size=self.image_size)

        elif self.augmentation == 'jitter_astro':
            augmentation = Jitter_astro(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v2':
            augmentation = Jitter_astro_v2(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v3':
            augmentation = Jitter_astro_v3(size=self.image_size)

        elif self.augmentation == 'jitter_simclr':
            augmentation = Jitter_simclr(size=self.image_size)

        elif self.augmentation == 'crop_astro':
            augmentation = Crop_astro(size=self.image_size)

        elif self.augmentation == 'crop_simclr':
            augmentation = Crop_simclr(size=self.image_size)
            
        elif self.augmentation == 'rotation':
            augmentation = Rotation(size=self.image_size)

        elif self.augmentation == 'rotation_v2':
            augmentation = Rotation_v2(size=self.image_size)

        elif self.augmentation == 'rotation_v3':
            augmentation = Rotation_v3(size=self.image_size)

        elif self.augmentation == 'crop_rotation':
            augmentation = Center_crop_rotation(size=self.image_size)

        elif self.augmentation == 'blur':
            augmentation = Gaussian_blur(size=self.image_size)

        elif self.augmentation == 'perspective':
            augmentation = RandomPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective':
            augmentation = RotationPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective_blur':
            augmentation = RotationPerspectiveBlur(size=self.image_size)

        elif self.augmentation == 'grid_distortion':
            augmentation = GridDistortion(size=self.image_size)

        elif self.augmentation == 'rot_grid':
            augmentation = RotationGrid(size=self.image_size)

        elif self.augmentation == 'rot_grid_blur':
            augmentation = RotationGridBlur(size=self.image_size)

        elif self.augmentation == 'elastic_transform':
            augmentation = ElasticTransform(size=self.image_size)

        elif self.augmentation == 'rot_elastic':
            augmentation = RotationElastic(size=self.image_size)

        elif self.augmentation == 'rot_elastic_blur':
            augmentation = RotationElasticBlur(size=self.image_size)

        elif self.augmentation == 'elastic_grid':
            augmentation = ElasticGrid(size=self.image_size)

        elif self.augmentation == 'elastic_prespective':
            augmentation = ElasticPerspective(size=self.image_size)

        elif self.augmentation == 'grid_perspective':
            augmentation = GridPerspective(size=self.image_size)

        elif self.augmentation == 'rot_elastic_grid_perspective':
            augmentation = RotElasticGridPerspective(size=self.image_size)

        elif self.augmentation == 'without_aug':
            augmentation = None


        # Data reading
        self.training_data = Dataset_stamps_v2(
                            self.data_path,
                            'Train',
                            image_size=self.image_size,
                            image_transformation=augmentation,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps_v2(
                            self.data_path,
                            'Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps_v2(
                            self.data_path,
                            'Test',
                            image_size=self.image_size,
                            image_transformation=None,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        return None


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
            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)#, betas=(0.5, 0.9))

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)#, betas=(0.5, 0.9))

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
            augmentation,
            projection_dim,
            temperature,
            lr_encoder,
            batch_size_encoder,
            optimizer_encoder,
            data_path_simclr='dataset/td_ztf_stamp_simclr_300.pkl',
            data_path_classifier='dataset/td_ztf_stamp_17_06_20.pkl'):

        super().__init__()
        

        # Hyperparameters of class are saved
        self.encoder_name = encoder_name
        self.method = method
        self.image_size = image_size
        self.augmentation = augmentation
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.lr_encoder = lr_encoder
        self.batch_size_encoder = batch_size_encoder
        self.optimizer_encoder = optimizer_encoder
        self.data_path_simclr = data_path_simclr
        self.data_path_classifier = data_path_classifier


        # Encoder loss
        if (self.method=='supcon'):
            self.criterion_encoder = SupConLoss(self.temperature)

        elif (self.method == 'simclr'):
            self.criterion_encoder = NT_Xent(self.batch_size_encoder, self.temperature)


        if (self.encoder_name == 'pstamps'):

            # Initialize P-stamp network
            self.encoder = P_stamps_net_SimCLR_v2(k=1.0)

            # Output features of the encoder
            self.n_features_encoder = self.encoder.bn1.num_features


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


        elif (self.encoder_name == 'resnet152'):

            # Initialize resnet152
            self.encoder = torchvision.models.resnet152()

            # Output features of the encoder
            self.n_features_encoder = self.encoder.fc.in_features

            # Last layer of the encoder is removed
            self.encoder.fc = Identity()


        # Initialize SimCLR network
        self.CLR = CLR(
            encoder=self.encoder,
            input_size=self.n_features_encoder,
            projection_size=self.projection_dim)


        # Save hyperparameters
        self.save_hyperparameters()


    def forward(self, x_img_i, x_img_j):

        # Forward of SimCLR 
        h_i, h_j, z_i, z_j = self.CLR(x_img_i, x_img_j)

        return h_i, h_j, z_i, z_j


    def output_proyection(self, x_img_i, x_img_j):

        # Forward of SimCLR 
        _, _, z_i, z_j = self.CLR(x_img_i, x_img_j)

        return (z_i, z_j)


    # Training loop
    def training_step(self, batch, batch_idx):

        self.CLR.train()

        return self.prediction_target_loss(batch, batch_idx)


    # End of training loop
    def training_epoch_end(self, outputs):

        loss_encoder = torch.stack([x['loss'] for x in outputs]).mean()
        pos = torch.stack([x['pos'] for x in outputs]).mean()
        neg = torch.stack([x['neg'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss_encoder/Train", loss_encoder, self.current_epoch)
        self.log('loss_train_enc', loss_encoder, logger=False, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        self.log('positive_mean', pos, prog_bar=False)
        self.log('negative_mean', neg, prog_bar=False)
        return None


    # Compute y_pred, y_true and loss for the batch
    def prediction_target_loss(self, batch, batch_idx):

        x_img_i, x_img_j = batch
        (z_i, z_j) = self.output_proyection(x_img_i, x_img_j)

        # Compute loss of encoder
        loss_encoder, pos, neg = self.criterion_encoder(z_i, z_j)

        return {'loss': loss_encoder, 'pos': pos, 'neg': neg}


    def visualization(self, dataset, file):

        # Load dataset
        if (dataset=='Validation'):
            dataloader = self.val_dataloader()

        elif (dataset=='Test'):
            dataloader = self.test_dataloader()

        # evaluation mode
        self.CLR.eval()

        # save y_true and y_pred of dataloader
        outputs = []

        # Iterate dataloader
        for idx, batch in enumerate(dataloader):

            # Evaluation loop
            x_img, x_feat, y_true = batch

            # Evaluate batch
            with torch.no_grad():
                h = self.CLR.encoder(x_img)
                h = h * (1/torch.norm(h, dim=1, keepdim=True))
                z = self.CLR.projector(h)
                z = z * (1/torch.norm(z, dim=1, keepdim=True))

            # Save output
            outputs.append({'y_true': y_true, 'h': h, 'z': z})

        # Concatenate results
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        h = torch.cat([x['h'] for x in outputs], dim=0)
        z = torch.cat([x['z'] for x in outputs], dim=0)

        # Plot confusion matrix for dataset
        title = f'Visualization of feature space ({self.augmentation}) - {dataset}'

        class_colours = ["green", "gray", "brown", "blue", "red"]
        class_labels = ['AGN', 'SN', 'VS', 'Asteroid', 'Bogus']
        class_instances = {}

        for i, label in enumerate(class_labels):
            class_instances[label] = np.where(y_true == i)[0]

        tsne_m = TSNE(n_components=2, n_jobs=8, random_state=42)
        h_embedded = tsne_m.fit_transform(h)
        z_embedded = tsne_m.fit_transform(z)
        #h_embedded = UMAP(n_neighbors=5, random_state=42).fit_transform(h)
        #z_embedded = UMAP(n_neighbors=5, random_state=42).fit_transform(z)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        # Plot
        for (label, colour) in zip(class_labels, class_colours):
            ax[0].scatter(h_embedded[class_instances[label], 0], h_embedded[class_instances[label], 1], c=colour, s=10)
            ax[1].scatter(z_embedded[class_instances[label], 0], z_embedded[class_instances[label], 1], c=colour, s=10)

        fig.legend(
            bbox_to_anchor=(1, 0.13),
            loc="lower left",
            ncol=1,
            labels=class_labels)

        ax[0].set_title("Encoder output")
        ax[0].set_xlabel("$x_{0}$")
        ax[0].set_ylabel("$x_{1}$")
        ax[1].set_title("Projector output")
        ax[1].set_xlabel("$x_{0}$")
        ax[1].set_ylabel("$x_{1}$")
        plt.suptitle(title)

        plt.tight_layout()
        plt.savefig(file, bbox_inches="tight", dpi=300)

        return None


    # Prepare datasets
    def prepare_data(self):

        if self.augmentation == 'simclr':
            augmentation = SimCLR_augmentation(size=self.image_size)

        if self.augmentation == 'simclr2':
            augmentation = SimCLR_augmentation_v2(size=self.image_size)

        if self.augmentation == 'simclr3':
            augmentation = SimCLR_augmentation_v3(size=self.image_size)

        elif self.augmentation == 'astro':
            augmentation = Astro_augmentation(size=self.image_size)

        elif self.augmentation == 'astro0':
            augmentation = Astro_augmentation_v0(size=self.image_size)

        elif self.augmentation == 'astro2':
            augmentation = Astro_augmentation_v2(size=self.image_size)

        elif self.augmentation == 'astro3':
            augmentation = Astro_augmentation_v3(size=self.image_size)

        elif self.augmentation == 'astro4':
            augmentation = Astro_augmentation_v4(size=self.image_size)

        elif self.augmentation == 'astro5':
            augmentation = Astro_augmentation_v5(size=self.image_size)

        elif self.augmentation == 'astro6':
            augmentation = Astro_augmentation_v6(size=self.image_size)

        elif self.augmentation == 'astro7':
            augmentation = Astro_augmentation_v7(size=self.image_size)

        elif self.augmentation == 'astro8':
            augmentation = Astro_augmentation_v8(size=self.image_size)

        elif self.augmentation == 'astro9':
            augmentation = Astro_augmentation_v9(size=self.image_size)

        elif self.augmentation == 'jitter_astro':
            augmentation = Jitter_astro(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v2':
            augmentation = Jitter_astro_v2(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v3':
            augmentation = Jitter_astro_v3(size=self.image_size)

        elif self.augmentation == 'jitter_simclr':
            augmentation = Jitter_simclr(size=self.image_size)

        elif self.augmentation == 'crop_astro':
            augmentation = Crop_astro(size=self.image_size)

        elif self.augmentation == 'crop_simclr':
            augmentation = Crop_simclr(size=self.image_size)
            
        elif self.augmentation == 'rotation':
            augmentation = Rotation(size=self.image_size)

        elif self.augmentation == 'rotation_v2':
            augmentation = Rotation_v2(size=self.image_size)

        elif self.augmentation == 'rotation_v3':
            augmentation = Rotation_v3(size=self.image_size)

        elif self.augmentation == 'blur':
            augmentation = Gaussian_blur(size=self.image_size)

        elif self.augmentation == 'perspective':
            augmentation = RandomPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective':
            augmentation = RotationPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective_blur':
            augmentation = RotationPerspectiveBlur(size=self.image_size)

        elif self.augmentation == 'grid_distortion':
            augmentation = GridDistortion(size=self.image_size)

        elif self.augmentation == 'rot_grid':
            augmentation = RotationGrid(size=self.image_size)

        elif self.augmentation == 'rot_grid_blur':
            augmentation = RotationGridBlur(size=self.image_size)

        elif self.augmentation == 'elastic_transform':
            augmentation = ElasticTransform(size=self.image_size)

        elif self.augmentation == 'rot_elastic':
            augmentation = RotationElastic(size=self.image_size)

        elif self.augmentation == 'rot_elastic_blur':
            augmentation = RotationElasticBlur(size=self.image_size)

        elif self.augmentation == 'elastic_grid':
            augmentation = ElasticGrid(size=self.image_size)

        elif self.augmentation == 'elastic_prespective':
            augmentation = ElasticPerspective(size=self.image_size)

        elif self.augmentation == 'grid_perspective':
            augmentation = GridPerspective(size=self.image_size)

        elif self.augmentation == 'rot_elastic_grid_perspective':
            augmentation = RotElasticGridPerspective(size=self.image_size)


        # Data reading -- Train
        self.training_data_aug = Dataset_simclr(
            self.data_path_simclr,
            dataset='Train',
            image_size=self.image_size,
            image_transformation=augmentation)


    # Prepare datasets
    def prepare_data_fast(self):

        # Data reading --Validation
        self.validation_data_aug = Dataset_stamps_v2(
            self.data_path_classifier,
            dataset='Validation',
            image_size=self.image_size,
            image_transformation=None,
            image_original_and_augmentated=False,
            one_hot_encoding=False,
            discarted_features=[13,14,15])


        # Data reading -- Test
        self.test_data_aug = Dataset_stamps_v2(
            self.data_path_classifier,
            dataset='Test',
            image_size=self.image_size,
            image_transformation=None,
            image_original_and_augmentated=False,
            one_hot_encoding=False,
            discarted_features=[13,14,15])


    def train_dataloader(self):

        # Data loader -- encoder
        train_dataloader_encoder = DataLoader(
            self.training_data_aug,
            batch_size=self.batch_size_encoder,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=False,
            drop_last=True)

        return train_dataloader_encoder


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

        if self.optimizer_encoder == "LARS":            
            optimizer_encoder = LARS(self.CLR.parameters(), lr=self.lr_encoder, weight_decay=0.0, momentum=0.9, use_nesterov=True)


        aux_scheduler = CosineLRScheduler(optimizer_encoder,
                                          warmup_t=5,
                                          warmup_lr_init=self.lr_encoder/3,
                                          t_initial=850,
                                          cycle_limit=1,
                                          lr_min=0.0001)

        def lr_lambda(epoch): return aux_scheduler.get_epoch_values(epoch)[0] / self.lr_encoder

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_encoder, lr_lambda=lr_lambda),
            'name': 'lr_cur'
            }

        return [optimizer_encoder], [lr_scheduler]

# -----------------------------------------------------------------------------


class Fine_SimCLR(pl.LightningModule):

    def __init__(
            self,
            simclr_model,
            batch_size,
            drop_rate,
            beta_loss,
            lr,
            optimizer,
            with_features=True,
            augmentation='without_aug',
            data_path='dataset/td_ztf_stamp_17_06_20.pkl'):
        super().__init__()


        # Hyperparameters of class are saved
        self.image_size = simclr_model.image_size
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer = optimizer
        self.with_features = with_features
        self.augmentation = augmentation
        self.data_path = data_path


        # dataset features are included to predict
        n_features_dataset = 23 if with_features else 0

        # Number of features (linear classifier)
        n_features = simclr_model.n_features_encoder + n_features_dataset


        # Model
        if (simclr_model.encoder_name == 'pstamps'):
            #if (with_features):
            self.model = torch.nn.Sequential(
                simclr_model.CLR.encoder,
                Three_layers_classifier(
                    input_size_f1=n_features,
                    input_size_f2=64,
                    input_size_f3=64,
                    n_classes=5,
                    drop_rate=drop_rate)
                )

            #else:
            #    self.model = torch.nn.Sequential(
            #        simclr_model.CLR.encoder,
            #        simclr_model.classifier)


        elif ('resnet' in simclr_model.encoder_name):
            self.model = torch.nn.Sequential(
                simclr_model.CLR.encoder,
                Linear_classifier(
                    input_size=n_features,
                    n_classes=5)
            )



        self.save_hyperparameters(
            "batch_size",
            "drop_rate",
            "beta_loss",
            "lr",
            "optimizer",
            "with_features",
            "augmentation"
        )


    def forward(self, x_img, x_feat):

        # The encoder is used to compute the features
        h = self.model[0](x_img)

        # Features computed from image and features of dataset are concatenated
        if (self.with_features): x = torch.cat((h, x_feat), dim=1)

        # Features of dataset are not used
        else: x = h

        logits = self.model[1](x)

        return logits


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
        self.log('accuracy_val', metrics['Accuracy'], logger=False, prog_bar=True)
        self.log('loss_val', avg_loss, logger=False, prog_bar=True)

        #cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #self.log("lr", cur_lr, prog_bar=True)

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

        logits = self.forward(x_img, x_feat)
        loss = self.criterion(logits, y_true)
        y_pred = logits.softmax(dim=1)

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
                logits = self.forward(x_img, x_feat)
                y_pred = logits.softmax(dim=1)

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


    # Prepare datasets
    def prepare_data(self):

        if self.augmentation == 'simclr':
            augmentation = SimCLR_augmentation(size=self.image_size)

        elif self.augmentation == 'astro':
            augmentation = Astro_augmentation(size=self.image_size)

        elif self.augmentation == 'jitter_astro':
            augmentation = Jitter_astro(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v2':
            augmentation = Jitter_astro_v2(size=self.image_size)

        elif self.augmentation == 'jitter_astro_v3':
            augmentation = Jitter_astro_v3(size=self.image_size)

        elif self.augmentation == 'jitter_simclr':
            augmentation = Jitter_simclr(size=self.image_size)

        elif self.augmentation == 'crop_astro':
            augmentation = Crop_astro(size=self.image_size)

        elif self.augmentation == 'crop_simclr':
            augmentation = Crop_simclr(size=self.image_size)
            
        elif self.augmentation == 'rotation':
            augmentation = Rotation(size=self.image_size)

        elif self.augmentation == 'rotation_v2':
            augmentation = Rotation_v2(size=self.image_size)

        elif self.augmentation == 'rotation_v3':
            augmentation = Rotation_v3(size=self.image_size)

        elif self.augmentation == 'crop_rotation':
            augmentation = Center_crop_rotation(size=self.image_size)

        elif self.augmentation == 'blur':
            augmentation = Gaussian_blur(size=self.image_size)

        elif self.augmentation == 'perspective':
            augmentation = RandomPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective':
            augmentation = RotationPerspective(size=self.image_size)

        elif self.augmentation == 'rot_perspective_blur':
            augmentation = RotationPerspectiveBlur(size=self.image_size)

        elif self.augmentation == 'grid_distortion':
            augmentation = GridDistortion(size=self.image_size)

        elif self.augmentation == 'rot_grid':
            augmentation = RotationGrid(size=self.image_size)

        elif self.augmentation == 'rot_grid_blur':
            augmentation = RotationGridBlur(size=self.image_size)

        elif self.augmentation == 'elastic_transform':
            augmentation = ElasticTransform(size=self.image_size)

        elif self.augmentation == 'rot_elastic':
            augmentation = RotationElastic(size=self.image_size)

        elif self.augmentation == 'rot_elastic_blur':
            augmentation = RotationElasticBlur(size=self.image_size)

        elif self.augmentation == 'elastic_grid':
            augmentation = ElasticGrid(size=self.image_size)

        elif self.augmentation == 'elastic_prespective':
            augmentation = ElasticPerspective(size=self.image_size)

        elif self.augmentation == 'grid_perspective':
            augmentation = GridPerspective(size=self.image_size)

        elif self.augmentation == 'rot_elastic_grid_perspective':
            augmentation = RotElasticGridPerspective(size=self.image_size)

        elif self.augmentation == 'without_aug':
            augmentation = None


        # Data reading
        self.training_data = Dataset_stamps_v2(
                            self.data_path,
                            dataset='Train',
                            image_size=self.image_size,
                            image_transformation=augmentation,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.validation_data = Dataset_stamps_v2(
                            self.data_path,
                            dataset='Validation',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        # Data reading
        self.test_data = Dataset_stamps_v2(
                            self.data_path,
                            dataset='Test',
                            image_size=self.image_size,
                            image_transformation=None,
                            image_original_and_augmentated=False,
                            one_hot_encoding=True,
                            discarted_features=[13,14,15])

        return None


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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1)

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        elif self.optimizer == "LARS":            
            optimizer = pl_bolts.optimizers.LARS(self.model.parameters(), lr=self.lr)

        return optimizer

# -----------------------------------------------------------------------------
