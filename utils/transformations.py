import torch
import torchvision
import cv2

from utils.random_resized_center_crop import RandomResizedCenterCrop
from utils.gaussian_blur import GaussianBlur

import albumentations as al
import numpy  as np

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
                #torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Apply gray scale
                torchvision.transforms.RandomGrayscale(p=0.2),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

                # Random rotation
                #Random_rotation()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation
class SimCLR_augmentation_v2:

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
                RandomResizedCenterCrop(scale=(0.08, 1.0), size=size),

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

# SimCLR augmentation
class SimCLR_augmentation_v3:

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
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0)
        
        self.augmentation = torchvision.transforms.Compose([

                # Random crop and random aspect ratio is applied. This crop is
                # finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.08, 1.0), size=size),

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

# SimCLR augmentation for astronomy
class Astro_augmentation_v0:

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
                RandomResizedCenterCrop(scale=(0.18, 1.0), size=size),

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

# SimCLR augmentation for astronomy
class Astro_augmentation_v2:

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

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v3:

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
        
        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),


            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v4:

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
        
        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),
                
                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),


            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v5:

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
        
        new_transform = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC),
        
                                   al.GridDistortion(p=0.3,
                                                     num_steps=5,
                                                     distort_limit=0.3,
                                                     border_mode=cv2.BORDER_CONSTANT,
                                                     interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: new_transform(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),
                
                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),


            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v6:

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
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

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

# SimCLR augmentation for astronomy
class Astro_augmentation_v7:

    """
    A stochastic data augmentation module for astronomy that transforms any
    given data example randomly resulting in two correlated views of the same
    example, denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Apply color transformation
                torchvision.transforms.RandomApply([color_jitter], p=0.8),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v8:

    """
    A stochastic data augmentation module for astronomy that transforms any
    given data example randomly resulting in two correlated views of the same
    example, denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

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

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

# SimCLR augmentation for astronomy
class Astro_augmentation_v9:

    """
    A stochastic data augmentation module for astronomy that transforms any
    given data example randomly resulting in two correlated views of the same
    example, denoted x_i and x_j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 1

        # Color transformation: brightness, contrast, saturation ,hue
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        new_transform = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC),
        
                                   al.GridDistortion(p=0.3,
                                                     num_steps=5,
                                                     distort_limit=0.3,
                                                     border_mode=cv2.BORDER_CONSTANT,
                                                     interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random center crop and random aspect ratio is applied. This
                # crop is finally resized to the given size.
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: new_transform(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

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

class Jitter_astro_v2:
    """
    A stochastic jitter augmentation.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 0.5

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

class Jitter_astro_v3:
    """
    A stochastic jitter augmentation.
    """

    def __init__(self, size):
        
        # Constant for color transformation
        s = 0.25

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
                RandomResizedCenterCrop(scale=(0.18, 0.5), size=size),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
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
                torchvision.transforms.RandomResizedCrop(scale=(0.18, 0.5), size=size),

                # Transform image to tensor
                torchvision.transforms.ToTensor()
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

class Rotation_v2:
    """
    A stochastic rotation augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=True),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),


                # Transform image to tensor
                torchvision.transforms.ToTensor()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class Rotation_v3:
    """
    A stochastic rotation augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),


                # Transform image to tensor
                torchvision.transforms.ToTensor()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

class Center_crop_rotation:
    """
    A stochastic centered crop and rotation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Center crop
                RandomResizedCenterCrop(scale=(0.1, 0.5), size=size),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Transform image to tensor
                torchvision.transforms.ToTensor()

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

class RandomPerspective:
    """
    A stochastic perspective augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotationPerspective:
    """
    A stochastic perspective augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotationPerspectiveBlur:
    """
    A stochastic perspective augmentation.
    """

    def __init__(self, size):
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class GridDistortion:
    """
    A stochastic grid distortion.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.5,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Transform image to tensor
                torchvision.transforms.ToTensor()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

class RotationGrid:
    """
    A stochastic perspective augmentation.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotationGridBlur:
    """
    A stochastic perspective augmentation.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Grid distortion
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class ElasticTransform:
    """
    A stochastic elastic transformation.
    """

    def __init__(self, size):

        elastic_trans = al.Compose([al.ElasticTransform(p=0.5,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Transform image to tensor
                torchvision.transforms.ToTensor()

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotationElastic:
    """
    A stochastic rotation and elastic transformation.
    """

    def __init__(self, size):

        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotationElasticBlur:
    """
    A stochastic rotation, elastic transformation and blur.
    """

    def __init__(self, size):

        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])
        
        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Blur image
                torchvision.transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class ElasticGrid:
    """
    A stochastic elastic transformation and grid distortion.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])

        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class ElasticPerspective:
    """
    A stochastic elastic and perspective augmentation.
    """

    def __init__(self, size):

        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class GridPerspective:
    """
    A stochastic grid distortion and perspective augmentation.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RotElasticGridPerspective:
    """
    A stochastic rotation, elastic transformation, grid distortion and perspective augmentation.
    """

    def __init__(self, size):

        grid_dist = al.Compose([al.GridDistortion(p=0.3,
                                                  num_steps=5,
                                                  distort_limit=0.3,
                                                  border_mode=cv2.BORDER_CONSTANT,
                                                  interpolation=cv2.INTER_CUBIC)])

        elastic_trans = al.Compose([al.ElasticTransform(p=0.3,
                                                        alpha=50,
                                                        sigma=120 * 0.05,
                                                        alpha_affine=50 * 0.03,
                                                        border_mode=cv2.BORDER_CONSTANT,
                                                        interpolation=cv2.INTER_CUBIC)])

        self.augmentation = torchvision.transforms.Compose([

                # Random rotation
                torchvision.transforms.RandomRotation(180, expand=False),

                # Random horizontal flip
                torchvision.transforms.RandomHorizontalFlip(),

                # Center crop
                torchvision.transforms.CenterCrop(size=size),

                # Elastic transformation
                torchvision.transforms.Lambda(lambda x: np.array(x)),
                torchvision.transforms.Lambda(lambda x: elastic_trans(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: grid_dist(image=x)['image']),
                torchvision.transforms.Lambda(lambda x: np.uint8(np.round(x))),
                torchvision.transforms.ToPILImage(),

                # RandomPerspective
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RandomHorizontalFlip:
    """
    A stochastic rotation, elastic transformation, grid distortion and perspective augmentation.
    """

    def __init__(self, size):

        self.augmentation = torchvision.transforms.Compose([

                torchvision.transforms.CenterCrop(size=size),

                torchvision.transforms.RandomHorizontalFlip(),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)

# ------------------------------------------------------------------------------

class RandomGrayscale:
    """
    A stochastic rotation, elastic transformation, grid distortion and perspective augmentation.
    """

    def __init__(self, size):

        self.augmentation = torchvision.transforms.Compose([

                torchvision.transforms.CenterCrop(size=size),

                torchvision.transforms.RandomGrayscale(p=0.2),

                # Transform image to tensor
                torchvision.transforms.ToTensor(),

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
