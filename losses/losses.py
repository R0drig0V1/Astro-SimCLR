import torch
import torch.nn as nn

# -----------------------------------------------------------------------------

# P_stramps loss with cross-entropy (onehot encoding) and entropy
# regulatization
class P_stamps_loss(nn.Module):

    def __init__(self, batch_size, beta):
        super(P_stamps_loss, self).__init__()
        self.batch_size = batch_size
        self.beta = beta

    def forward(self, outputs, targets):

        N = self.batch_size

        loss = - torch.sum(targets * torch.log2(outputs))
        loss += self.beta * torch.sum(outputs * torch.log2(outputs))
        loss /= N

        return loss

# -----------------------------------------------------------------------------

# Spijkervet / SimCLR / simclr / modules / nt_xent.py
class NT_Xent(nn.Module):

    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)


    def mask_correlated_samples(self, batch_size):

        N = 2 * batch_size

        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):

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

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss

#------------------------------------------------------------------------------