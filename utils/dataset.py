import csv
import torch
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from utils.transformations import Resize_img

# -----------------------------------------------------------------------------

label_names = ['AGN', 'SN', 'VS', 'Asteroid', 'Bogus']

feature_names = ['sgscore1', 'distpsnr1', 'sgscore2', 'distpsnr2', 'sgscore3',
                 'distpsnr3', 'isdiffpos', 'fwhm', 'magpsf', 'sigmapsf', 'ra',
                 'dec', 'diffmaglim', 'rb', 'distnr', 'magnr', 'classtar',
                 'ndethist', 'ncovhist', 'ecl_lat', 'ecl_long', 'gal_lat',
                 'gal_long', 'non_detections', 'chinr', 'sharpnr']

# -----------------------------------------------------------------------------

# Change the format of image, from np.float [0,1] to np.int8 (0,255)
def img_float2int(img):

    img_255 = 255 * img
    img_round = np.uint8(np.round(img_255))

    return img_round

# -----------------------------------------------------------------------------

# One hot encoding transformation to use softmax
def one_hot_trans(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=5)

# -----------------------------------------------------------------------------

def repair_non_squared_stamp(stamp: np.ndarray,
                             xpos: int,
                             ypos: int) -> np.ndarray:

    """
    Fill with `np.nan` a misshaped stamps.
    """

    height, width, _ = stamp.shape
    
    if ((height == 63) & (width == 63)):
        return stamp
    
    if (height != 63):
        nan_pad = np.empty((63 - height, width, 3))
        nan_pad[:] = np.nan
        if ypos < 40.0:
            stamp = np.concatenate([nan_pad, stamp], axis=0) 
        else:
            stamp = np.concatenate([stamp, nan_pad], axis=0)
            
    if (width != 63):
        nan_pad = np.empty((63, 63 - width, 3))
        nan_pad[:] = np.nan
        if xpos < 40.0:
            stamp = np.concatenate([nan_pad, stamp], axis=1)
        else:
            stamp = np.concatenate([stamp, nan_pad], axis=1)
    return stamp

# -----------------------------------------------------------------------------

def preprocess_stamps(stamp: np.ndarray,
                      nan_val: float) -> np.ndarray:

    """
    Returns a preprocessed stamp; get a stamp with values between 0 and 1. Normalization.
    """
    
    stamp_image = np.nan_to_num(stamp, nan=np.nan, posinf=np.nan, neginf=np.nan)
    stamp_image = stamp_image - np.nanmin(stamp_image, axis=(0,1))
    stamp_image = stamp_image / (np.nanmax(stamp_image, axis=(0,1)) + 0.000001)
    stamp_image = np.nan_to_num(stamp_image, nan=nan_val, posinf=nan_val, neginf=nan_val)
    
    return stamp_image

# -----------------------------------------------------------------------------

# Class to load alerts of ZTF
class Dataset_stamps(torch.utils.data.Dataset):

    def __init__(
            self,
            pickle,
            dataset,
            image_size,
            image_transformation=None,
            image_original_and_augmentated=True,
            one_hot_encoding=True,
            discarted_features=[13,14,15]
            ):


        # Load parameters
        self.pickle = pickle
        self.dataset = dataset
        self.additional_transformation = image_transformation is not None
        self.image_original_and_augmentated = image_original_and_augmentated
        self.discarted_features = discarted_features

        # Mean and std to normalize
        #self.mean_features = np.mean(np.array(self.pickle['Train']['features'], dtype=np.float32), axis=0)
        #self.std_features = np.std(np.array(self.pickle['Train']['features'], dtype=np.float32), axis=0)

        # Base transformation for images
        self.base_transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                Resize_img(size=image_size)
                ])


        # Additional transformation
        if (self.additional_transformation):

            self.transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                image_transformation
                ])


        # Apply transformation for labels
        if (one_hot_encoding):
            self.target_transform = transforms.Compose([transforms.Lambda(one_hot_trans)])

        else:
            self.target_transform = transforms.Compose([])


    def __len__(self):
        return len(self.pickle[self.dataset]['labels'])


    def __getitem__(self, idx):

        # Get items
        image = self.pickle[self.dataset]['images'][idx]

        # Get features
        feature_numpy = (np.array(self.pickle[self.dataset]['features'][idx],
                                 dtype=np.float32))

        # Normalization
        #feature_numpy = (feature_numpy - self.mean_features) / self.std_features

        # Features are deleted and converted from numpy to torch
        feature = torch.from_numpy(np.delete(feature_numpy, self.discarted_features))

        # Get labels
        label = self.pickle[self.dataset]['labels'][idx]

        # Base transformation for images is applied
        image_b = self.base_transformation(image)

        # Transformation for labels
        label = self.target_transform(label)


        # Batch for simultaneous training of encoder and classifier
        if (self.additional_transformation and self.image_original_and_augmentated):
            image_t = self.transformation(image)
            return image_b, image_t, feature, label

        # Batch for stamps classifier with augmentations
        elif (self.additional_transformation and self.image_original_and_augmentated==False):
            image_t = self.transformation(image)[0]
            return image_t, feature, label

        # Batch for encoder training
        else:
            return image_b, feature, label

# -----------------------------------------------------------------------------

# Class to load alerts of ZTF
class Dataset_stamps_v2(torch.utils.data.Dataset):

    def __init__(
            self,
            path,
            dataset,
            image_size,
            image_transformation=None,
            image_original_and_augmentated=True,
            one_hot_encoding=True,
            discarted_features=[13,14,15]
            ):


        # Load parameters
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)[dataset]


        self.additional_transformation = image_transformation is not None
        self.image_original_and_augmentated = image_original_and_augmentated
        self.discarted_features = discarted_features

        # Mean and std to normalize
        #self.mean_features = np.mean(np.array(self.dataset['features'], dtype=np.float32), axis=0)
        #self.std_features = np.std(np.array(self.dataset['features'], dtype=np.float32), axis=0)

        # Base transformation for images
        self.base_transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                Resize_img(size=image_size)
                ])


        # Additional transformation
        if (self.additional_transformation):

            self.transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                image_transformation
                ])


        # Apply transformation for labels
        if (one_hot_encoding):
            self.target_transform = transforms.Compose([transforms.Lambda(one_hot_trans)])

        else:
            self.target_transform = transforms.Compose([])


    def __len__(self):
        return len(self.dataset['labels'])


    def __getitem__(self, idx):

        # Get items
        image = self.dataset['images'][idx]

        # Get features
        feature_numpy = (np.array(self.dataset['features'][idx],
                                 dtype=np.float32))

        # Normalization
        #feature_numpy = (feature_numpy - self.mean_features) / self.std_features

        # Features are deleted and converted from numpy to torch
        feature = torch.from_numpy(np.delete(feature_numpy, self.discarted_features))

        # Get labels
        label = self.dataset['labels'][idx]

        # Base transformation for images is applied
        image_b = self.base_transformation(image)

        # Transformation for labels
        label = self.target_transform(label)


        # Batch for simultaneous training of encoder and classifier
        if (self.additional_transformation and self.image_original_and_augmentated):
            image_t = self.transformation(image)
            return image_b, image_t, feature, label

        # Batch for stamps classifier with augmentations
        elif (self.additional_transformation and self.image_original_and_augmentated==False):
            image_t = self.transformation(image)[0]
            return image_t, feature, label

        # Batch for encoder training
        else:
            return image_b, feature, label


# -----------------------------------------------------------------------------

# Class to load alerts of ZTF
class Dataset_simclr(torch.utils.data.Dataset):

    def __init__(
            self,
            path,
            dataset,
            image_size,
            image_transformation=None):


        # Load parameters
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)[dataset]


        self.additional_transformation = image_transformation is not None


        # Base transformation for images
        self.base_transformation = transforms.Compose([
            transforms.Lambda(img_float2int),
            transforms.ToPILImage(),
            Resize_img(size=image_size)
            ])


        # Additional transformation
        if (self.additional_transformation):

            self.transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                image_transformation
                ])


    def __len__(self):
        return len(self.dataset['images'])


    def __getitem__(self, idx):

        # Get items
        image = self.dataset['images'][idx]

        return self.transformation(image)


# -----------------------------------------------------------------------------

#https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198

class BalancedBatchSampler(BatchSampler):

    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within
    these classes samples n_samples. Return batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):

        loader = DataLoader(dataset)

        self.labels_list = []

        for _, (_, _, label) in enumerate(loader):
            self.labels_list.append(int(torch.argmax(label)))

        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes


    def __iter__(self):

        self.count = 0

        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)

            indices = []
            for class_ in classes:

                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.n_classes * self.n_samples


    def __len__(self):
        return len(self.dataset) // self.batch_size

# -----------------------------------------------------------------------------


class Batch_sampler_step(BatchSampler):

    def __init__(self, n_data, steps, batch_size):

    
        self.n_data = n_data
        self.steps = steps
        self.batch_size = batch_size

        self.aux_data_loader = DataLoader(torch.arange(n_data), batch_size=batch_size, shuffle=True, drop_last=True)


    def __iter__(self):

        self.s = 0

        while (self.s < self.steps):

            for batch in self.aux_data_loader:

                yield batch
                self.s += 1

                if (self.s == self.steps):
                    break


    def __len__(self):
        return self.steps

# -----------------------------------------------------------------------------
