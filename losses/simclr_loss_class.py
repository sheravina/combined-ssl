import torch
import torch.nn as nn


class SimCLRLoss(nn.Module):
    """
    SimCLR loss function as a torch.nn.Module.

    Args:
        temperature (float): Temperature parameter for scaling
    """

    def __init__(self, temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Hidden feature representation of shape [b, 2, dim]

        Returns:
            torch.Tensor: Loss computed according to SimCLR
        """
        # Get shape information
        b, n, dim = features.size()
        assert n == 2, "SimCLR requires exactly 2 views per sample"

        # Create identity matrix for positive pairs
        mask = torch.eye(b, dtype=torch.float32).to(features.device)

        # Concatenate all features
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Compute dot product similarity
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        # Create mask for valid pairs
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(b).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive pairs
        loss = -((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss
