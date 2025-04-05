import torch
import torch.nn as nn


class SupervisedModel(nn.Module):
    def __init__(
        self,
        base_encoder,
        input_shape=None,
        feature_size=None,
        output_shape=None,
    ):
        super().__init__()

        self.encoder = base_encoder

        if feature_size is not None:
            self.feature_size = feature_size
        elif input_shape is not None:
            self.feature_size = self.encoder.calc_feat_size(input_shape)
        else:
            raise ValueError("Either feature_dim or input_shape must be provided")

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.feature_size, out_features=output_shape),
        )

    def forward(self, x):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)

        # Process through encoder
        x = self.encoder(x)
        # Process through classifier head
        x = self.classifier_head(x)
        return x
