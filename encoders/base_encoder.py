import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Base class for all encoder backbones."""

    @abstractmethod
    def forward(self, x):
        """Forward pass through the encoder."""
        pass

    def calc_feat_size(self, input_shape):
        """Calculate total feature size when flattened."""
        x_dim = torch.zeros(2, *input_shape, device=next(self.parameters()).device)
        with torch.no_grad():
            features = self.forward(x_dim)
            feature_size = features.shape[1]
        return feature_size