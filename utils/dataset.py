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
class Dataset_stamps_v2(torch.utils.data.Dataset):

    def __init__(
            self,
            path,
            dataset,
            image_size,
            image_transformation=None,
            one_hot_encoding=True,
            discarted_features=[13,14,15]
            ):


        # Load parameters
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)[dataset]

        self.discarted_features = discarted_features

        # Mean and std to normalize
        #self.mean_features = np.mean(np.array(self.dataset['features'], dtype=np.float32), axis=0)
        #self.std_features = np.std(np.array(self.dataset['features'], dtype=np.float32), axis=0)

        if image_transformation:
            self.transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                image_transformation.augmentation
            ])

        else: 
            self.transformation = transforms.Compose([
                transforms.Lambda(img_float2int),
                transforms.ToPILImage(),
                Resize_img(size=image_size)
            ])

        # Apply transformation for labels
        if (one_hot_encoding):
            self.dataset['labels'] = one_hot_trans(self.dataset['labels'])

        else:
            self.dataset['labels'] = torch.tensor(self.dataset['labels'])

        self.dataset['features'] = torch.from_numpy(np.delete(np.array(self.dataset['features'], dtype=np.float32), self.discarted_features, axis=1))


    def __len__(self):
        return len(self.dataset['labels'])


    def __getitem__(self, idx):

        # Get images
        image = self.dataset['images'][idx]

        # Get labels
        label = self.dataset['labels'][idx]

        # Get features
        feature = self.dataset['features'][idx]

        # Normalization
        #feature_numpy = (feature_numpy - self.mean_features) / self.std_features

        return self.transformation(image), feature, label

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
            self.dataset = pickle.load(f)[dataset]['images']


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
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.transformation(self.dataset[idx])


# -----------------------------------------------------------------------------
