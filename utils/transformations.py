import torch
import torchvision

from utils.random_resized_center_crop import RandomResizedCenterCrop
from utils.gaussian_blur import GaussianBlur

# ------------------------------------------------------------------------------

class Random_rotation:

    """
    Random rotation 
    """

    def __init__(self):
        pass

    def __call__(self, x):

        # 90 degree rotation applied
        times = torch.randint(4, (1,))[0]

        return torch.rot90(x, times, [1, 2])


# ------------------------------------------------------------------------------

# SimCLR augmentation
class SimCLR_augmentation:

    """
    A stochastic data augmentation module that transforms any given data
    example randomly resulting in two correlated views of the same example,
    denoted x_i and x_j, which we consider as a positive pair.

    GitHub: SimCLR / simclr / modules / transformations / simclr.py

    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.augmentation = torchvision.transforms.Compose([

                # Random crop and random aspect ratio is applied. This crop is
                # finally resized to the given size.
                torchvision.transforms.RandomResizedCrop(scale=(0.08, 1.0), size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Apply gray scale
                torchvision.transforms.RandomGrayscale(p=0.2),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

                # Random rotation
                Random_rotation()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation:

    """
    A stochastic data augmentation module for astronomy that transforms any
    given data example randomly resulting in two correlated views of the same
    example, denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0)
        
        self.augmentation = torchvision.transforms.Compose([

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.08, 1.0), size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

                # Random rotation
                Random_rotation()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Jitter_astro:
    """
    A stochastic jitter augmentation.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0)
        
        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Jitter_simclr:
    """
    A stochastic jitter augmentation.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

class Crop_astro:
    """
    A stochastic center crop augmentation.
    """

    def __init__(self, size):
        

        self.augmentation = torchvision.transforms.Compose([

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.08, 1.0), size=size),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Crop_simclr:
    """
    A stochastic crop augmentation.
    """

    def __init__(self, size):
        

        self.augmentation = torchvision.transforms.Compose([

                # Random crop and random aspect ratio is applied. This crop is
                # finally resized to the given size.
                torchvision.transforms.RandomResizedCrop(scale=(0.08, 1.0), size=size),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Rotation:
    """
    A stochastic rotation augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

                # Random rotation
                Random_rotation()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Gaussian_blur:
    """
    A stochastic gaussian blur.
    """

    def __init__(self, size):
        
        # Constant for kernel
        s = 0.1
        
        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# Resize image
class Resize_img:

    def __init__(self, size):

        self.resize = torchvision.transforms.Compose([

                # Crop image
                torchvision.transforms.CenterCrop(size=size),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.resize(x)

# ------------------------------------------------------------------------------
