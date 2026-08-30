import torch
from torch import nn


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy plus squared earth-mover distance so errors are penalised by how many KL grades they miss by."""

    def __init__(self, weight=None, label_smoothing=0.1, emd_weight=1.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.emd_weight = emd_weight

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probabilities).scatter_(1, targets.unsqueeze(1), 1.0)
        squared_emd = torch.sum((probabilities.cumsum(dim=1) - one_hot.cumsum(dim=1)) ** 2, dim=1).mean()
        return self.cross_entropy(logits, targets) + self.emd_weight * squared_emd
