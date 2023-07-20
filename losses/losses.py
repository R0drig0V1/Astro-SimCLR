import torch
import torch.nn as nn

# -----------------------------------------------------------------------------

# P_stramps loss with cross-entropy (onehot encoding) and entropy regulatization

class P_stamps_loss(nn.Module):

    def __init__(self, batch_size, beta):
        super(P_stamps_loss, self).__init__()

        # Batch size
        self.batch_size = batch_size

        # Entropy regulatization constant
        self.beta = beta

        # loss
        self.loss = nn.CrossEntropyLoss()

        # logsoftmax
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, logits, targets):

        # compute log soft max
        log_soft = self.logsoftmax(logits)

        # Loss average
        loss = self.loss(logits, torch.argmax(targets, dim=1))
        loss -= self.beta * self.loss(logits, torch.exp(log_soft))

        return loss

# -----------------------------------------------------------------------------

# Github: Spijkervet / SimCLR / simclr / modules / nt_xent.py

class NT_Xent(nn.Module):

    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()

        # Batch size
        self.batch_size = batch_size
        
        # Temperature constant
        self.temperature = temperature

        # Mask for negative samples during training and validation/test stage
        self.mask = self.mask_negative_samples(batch_size)
        self.mask_val_test = self.mask_negative_samples(100)

        # Function used to compute the loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # Distance criterion between projection features
        self.similarity_f = nn.CosineSimilarity(dim=2)


    def mask_negative_samples(self, batch_size):

        """
        Mask for negative samples
        """

        # Number of samples per batch
        N = 2 * batch_size

        # Inicializes matrix
        mask = torch.ones((N, N), dtype=bool)

        # Fills diagonal with zeros (positive samples)
        mask = mask.fill_diagonal_(0)

        # Position of negatives are filled with zeros
        for i in range(batch_size):

            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask
        

    def forward(self, z_i, z_j, labels=None):

        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we
        treat the other 2(N − 1) augmented examples within a minibatch as
        negative examples.
        """

        # Number of samples
        N = 2 * z_i.size(0)

        # Mask used for negatives (training or validation/test)
        if (z_i.size(0)==self.batch_size):
            mask = self.mask
        else: 
            mask = self.mask_val_test
        
        # Concatenates features
        z = torch.cat((z_i, z_j), dim=0)

        #  Matrix of distance between samples
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Distance between i-th sample and its positive
        sim_i_j = torch.diag(sim, N//2)
        sim_j_i = torch.diag(sim, -N//2)

        # Distance between positives and negatives
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        # Zero labels because positives are the first sample
        labels = torch.zeros(N, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((torch.zeros_like(positive_samples), negative_samples), dim=1)

        # Computes mean loss
        loss = self.criterion(logits, labels) - torch.sum(positive_samples)
        loss /= N

        with torch.no_grad():
            neg = torch.mean(negative_samples) * self.temperature
            pos = torch.mean(positive_samples) * self.temperature

        return loss, pos, neg

#------------------------------------------------------------------------------
