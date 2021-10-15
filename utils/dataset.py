import csv
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

# -----------------------------------------------------------------------------

label_names = ['AGN', 'SN', 'VS', 'Asteroid', 'Bogus']

feature_names = ['sgscore_1', 'sgscore_2', 'sgscore_3', 'distpsnr_1',
                 'distpsnr_2', 'distpsnr_3', 'isdiffpos', 'fwhm', 'magpsf',
                 'sigmapsf', 'RA', 'DEC', 'diffmaglim', 'classtar', 'ndethist',
                 'ncovhist', 'chinr', 'sharpnr', 'Ecliptic coordinates RA',
                 'Ecliptic coordinates DEC', 'Galactic coordinates RA',
                 'Galactic coordinates DEC', 'approx non-detections']

# -----------------------------------------------------------------------------

# It Recursively prints the type of the objects inside the input.
def dataset_structure(data, space=''):

    # Iteration along dictionary
    for keys in data:

        # If the object is other dictionary
        if(type(data[keys]) is dict):

            print('{0}-{1}:'.format(space,keys))
            dataset_structure(data[keys], space + '\t')

        else:

            # Type of the object
            type_obj = type(data[keys]).__name__

            # Extracts an example of the elements into the object
            example = data[keys][0]
            type_example = type(example).__name__

            # If the element is an ndarray, its shape is printed
            if(type_example=='ndarray'):
                shape_example = str(np.shape(example))
            else:
                shape_example = 'Non-ndarray'

            # Prints the information (length and type) of the structure inside
            # the dictionary
            print("{0}-{1:<10}N={2:<10}Type:{3}".format(space, keys, len(data[keys]), type_obj), end="")
            print("\t-->\tType:{1:<10}Shape:{2:<10}".format(space, type_example, shape_example))

# -----------------------------------------------------------------------------

# Change the format of image, from np.float [0,1] to np.int8 (0,255)
def img_float2int(img):

    img_255 = 255 * img
    img_round = np.uint8(np.round(img_255))
    return img_round

# -----------------------------------------------------------------------------

# One hot encoding to use softmax
def one_hot_trans(x):

    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=5)

# -----------------------------------------------------------------------------

# Class to load stamps
class Dataset_stamps(torch.utils.data.Dataset):

    def __init__(self, pickle, dataset, transform=None, one_hot_encoding=True):

        self.pickle = pickle
        self.dataset = dataset

        if (transform is None):
            self.transform = transforms.Compose([transforms.Lambda(img_float2int),
                                                transforms.ToPILImage()
                                                ])
        else:
            self.transform = transforms.Compose([transforms.Lambda(img_float2int),
                                                transforms.ToPILImage(),
                                                transform])

        if (one_hot_encoding):
            self.target_transform = transforms.Compose([transforms.Lambda(one_hot_trans)])

        else:
            self.target_transform = transforms.Compose([])


    def __len__(self):
        return len(self.pickle[self.dataset]['labels'])


    def __getitem__(self, idx):

        # Get items
        image = self.pickle[self.dataset]['images'][idx]

        feature = torch.from_numpy(np.array(
            self.pickle[self.dataset]['features'][idx],
            dtype=np.float32
            ))

        label = self.pickle[self.dataset]['labels'][idx]

        # Transformations
        image = self.transform(image)
        label = self.target_transform(label)

        return image, feature, label


# -----------------------------------------------------------------------------

#https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198

class BalancedBatchSampler(BatchSampler):

    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
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