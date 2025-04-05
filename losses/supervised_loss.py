import torch
import torch.nn as nn


class SupervisedLoss(nn.Module):
    """
    Simple supervised loss using cross entropy.
    """

    def __init__(self):
        super(SupervisedLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Cross entropy loss
        """
        return self.criterion(logits, targets)
