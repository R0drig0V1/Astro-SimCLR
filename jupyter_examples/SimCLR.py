import torch
import torchvision

import torch.nn as nn
import torch.distributed as dist

#------------------------------------------------------------------------------

# Spijkervet / SimCLR / simclr / modules / gather.py
class GatherLayer(torch.autograd.Function):

    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

# loss.py
#------------------------------------------------------------------------------

# Spijkervet / SimCLR / simclr / modules / nt_xent.py
class NT_Xent(nn.Module):

    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we
        treat the other 2(N − 1) augmented examples within a minibatch as
        negative examples.
        """

        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N
        # examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

#------------------------------------------------------------------------------

#  Spijkervet / SimCLR / simclr / modules / identity.py
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


#  Spijkervet / SimCLR / simclr / simclr.py
class SimCLR(nn.Module):

    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016)
    to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the
    average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i, h_j, z_i, z_j


#------------------------------------------------------------------------------

# SimCLR / simclr / modules / resnet.py
def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")

    return resnets[name]

#------------------------------------------------------------------------------

# SimCLR / simclr / modules / transformations / simclr.py
class TransformsSimCLR:

    """
    A stochastic data augmentation module that transforms any given data
    example randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

#------------------------------------------------------------------------------


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)



import os

#------------------------------------------------------------------------------

# https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
class DictX(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

#------------------------------------------------------------------------------

import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule

#------------------------------------------------------------------------------

class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # Load params
        for key in args.keys():
            self.hparams[key] = args[key]

        # initialize ResNet
        self.encoder = get_resnet(self.hparams.resnet, pretrained=False)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch
        loss = self.forward(x_i, x_j)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):

        scheduler = None

        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


#args = DictX(yaml_config_hook("config.yaml"))

args = DictX({
# distributed training
'nodes': 1,
'gpus': 1, # I recommend always assigning 1 GPU to 1 node
'nr': 0, # machine nr. in node (0 -- nodes - 1)
'dataparallel': 0, # Use DataParallel instead of DistributedDataParallel
'workers': 4,
'dataset_dir': "./datasets",

# train options
'seed': 42, # sacred handles automatic seeding when passed in the config
'batch_size': 128,
'image_size': 50, #224
'start_epoch': 0,
'epochs': 50,
'dataset': "CIFAR10", # STL10
'pretrain': True, 

# model options
'resnet': "resnet18",
'projection_dim': 64, # "[...] to project the representation to a 128-dimensional latent space"

# loss options
'optimizer': "Adam", # or LARS (experimental)
'weight_decay': 1.0e-6, # "optimized using LARS [...] and weight decay of 10−6"
'temperature': 0.5, # see appendix B.7.: Optimal temperature under different batch sizes

# reload options
'model_path': "save", # set to the directory containing `checkpoint_##.tar` 
'epoch_num': 50, # set to checkpoint number
'reload': False,

# logistic regression options
'logistic_batch_size': 256,
'logistic_epochs': 500})

if args.dataset == "STL10":
    train_dataset = torchvision.datasets.STL10(
        args.dataset_dir,
        split="unlabeled",
        download=True,
        transform=TransformsSimCLR(size=args.image_size),
    )
elif args.dataset == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        download=True,
        transform=TransformsSimCLR(size=args.image_size),
    )
else:
    raise NotImplementedError
    
if args.gpus == 1:
    workers = args.workers
else:
    workers = 0

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=workers)

cl = ContrastiveLearning(args)
    
trainer = Trainer.from_argparse_args(args, gpus=0)
trainer.sync_batchnorm = True
trainer.fit(cl, train_loader)

