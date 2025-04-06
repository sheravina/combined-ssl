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
        x_dim = torch.zeros(1, *input_shape, device=next(self.parameters()).device)
        with torch.no_grad():
            features = self.forward(x_dim)
            feature_size = features.shape[1] * features.shape[2] * features.shape[3]
        return feature_size

    def calc_feat_dim(self, input_shape):
        """Calculate feature dimension (width/height)."""
        x_dim = torch.zeros(1, *input_shape, device=next(self.parameters()).device)
        with torch.no_grad():
            features = self.forward(x_dim)
            feature_dim = features.shape[3]
        return feature_dim
