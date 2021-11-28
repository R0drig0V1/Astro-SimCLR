import csv
import torch

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

# Dataset is loaded
#with open('dataset/td_ztf_stamp_17_06_20.pkl', 'rb') as f:
#    data = pickle.load(f)

# Mean and std to normalize
#mean_features = np.mean(np.array(features, dtype=np.float32), axis=0)
#std_features = np.std(np.array(features, dtype=np.float32), axis=0)


mean_features = np.array([-0.2873714, 0.11175089, -0.03527644, -0.00055827,
                          -0.00971938, 0.00157936, 0.5222817, 0.0454788,
                          0.02194207, -0.04955371, -0.02466081, 0.02335262,
                          -0.0385435, 0.01949287, 0.09752761, 0.14096506,
                          0.09265201, -0.10308222, 0.02861715, 0.00802673,
                          -0.02782944, -0.01558139, 0.00917627, 0.03677692,
                          0.33868203, 0.2191507], dtype=np.float32)

std_features = np.array([1.0836236, 1.0093966, 0.99515635, 0.9982329,
                         0.996767, 0.994433, 0.91449946, 0.9764465,
                         0.90470445, 0.964727,1.0097935, 0.9968355,
                         0.9727092, 1.0060825, 1.0007532,0.9736594,
                         0.95906055, 0.9262455, 1.0111927, 1.0210558,
                         1.0055873, 1.0252767, 0.97880936, 1.0133779,
                         1.332348, 1.254867], dtype=np.float32)

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
        feature_numpy = (feature_numpy - mean_features) / std_features

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
