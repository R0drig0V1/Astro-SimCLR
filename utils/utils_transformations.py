import torchvision

# ------------------------------------------------------------------------------

# SimCLR / simclr / modules / transformations / simclr.py
class Augmentation_SimCLR:

    """
    A stochastic data augmentation module that transforms any given data
    example randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.augmentation = torchvision.transforms.Compose([

                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


# ------------------------------------------------------------------------------

class Resize_img:


    def __init__(self, size):

        self.resize = torchvision.transforms.Compose([

                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor()
            ])


    def __call__(self, x):
        return self.resize(x)