import torch
import torch.nn as nn


class JigsawLoss(nn.Module):
    """
    Simple supervised loss using cross entropy.
    """

    def __init__(self):
        super(JigsawLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Cross entropy loss
        """
        return self.criterion(logits, targets)
