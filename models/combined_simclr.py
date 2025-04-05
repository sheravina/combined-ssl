import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedSimCLR(nn.Module):
    def __init__(
        self,
        base_encoder,
        input_shape=None,
        feature_size=None,
        feature_dim=None,
        output_shape=None,
        hidden_units=1024,
        proj_units=128,
    ):
        super().__init__()

        # Use encoder as base
        self.encoder = base_encoder

        # Get the output dimension from the last layer
        if feature_dim is not None:
            self.feature_dim = feature_dim
        elif input_shape is not None:
            self.feature_dim = self.encoder.calc_feat_dim(input_shape)
        else:
            raise ValueError("Either feature_dim or input_shape must be provided")

        if feature_size is not None:
            self.feature_size = feature_size
        elif input_shape is not None:
            self.feature_size = self.encoder.calc_feat_size(input_shape)
        else:
            raise ValueError("Either feature_size or input_shape must be provided")

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.feature_size, out_features=output_shape),
        )

        # Add a projection head (MLP with one hidden layer)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, proj_units),
        )

    def forward(self, x):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)

        # Get features from encoder
        h = self.encoder(x)

        # Project features to the contrastive embedding space
        c = self.classifier_head(h)
        z = self.projection_head(h)
        z = F.normalize(z, p=2, dim=1)
        return c, z
