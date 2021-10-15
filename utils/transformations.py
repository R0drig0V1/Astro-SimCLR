import torchvision

from utils.random_resized_center_crop import RandomResizedCenterCrop

# ------------------------------------------------------------------------------

# SimCLR augmentation
class Augmentation_SimCLR:

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

                # Transform image to tensor
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_Augmentation_SimCLR:

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
