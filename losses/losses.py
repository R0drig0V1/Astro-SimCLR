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

        loss = - torch.sum(targets*torch.log2(outputs))
        loss += self.beta * torch.sum(outputs*torch.log2(outputs))
        loss /= N

        return loss

# -----------------------------------------------------------------------------