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

    def forward(self, outputs, targets):

        # Number of samples per batch
        N = targets.size(0)

        # Loss average
        loss = - torch.sum(targets * torch.log(torch.clamp(outputs, min=1e-18)))
        loss += self.beta * torch.sum(outputs * torch.log(torch.clamp(outputs, min=1e-18)))
        loss /= N

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
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Computes mean loss
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

#------------------------------------------------------------------------------

# Github: HobbitLong / SupContrast / losses.py


class SupConLoss(nn.Module):

    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """

    def __init__(self, temperature):
        super(SupConLoss, self).__init__()

        # Temperature constant
        self.temperature = temperature

        # Distance criterion between projection features
        self.similarity_f = nn.CosineSimilarity(dim=2)


    def forward(self, z_i, z_j, labels):

        # Number of samples
        N = 2 * z_i.size(0)

        # Mask of positives (mask of labels)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Repeat the mask (mask of samples) with and without diagonal
        mask = mask.repeat(2, 2)

        # Compute distance between samples
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_max = sim_max.detach()
        sim = sim - sim_max

        # Exponencial similarity
        exp_sim = torch.exp(sim)

        # Mask of different elements than i-th sample
        mask_not_diag = torch.ones((N, N), device=mask.device)
        mask_not_diag.fill_diagonal_(0)

        # Mask without diagonal
        mask = mask * mask_not_diag

        # compute log_prob
        log_prob = sim - torch.log(torch.clamp((exp_sim * mask_not_diag).sum(1, keepdim=True), min=1e-18))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss

#------------------------------------------------------------------------------
