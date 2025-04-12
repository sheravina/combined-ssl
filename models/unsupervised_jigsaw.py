import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel  # Import BaseModel from the base module

class Jigsaw(BaseModel):
    def __init__(self, base_encoder, input_shape=None, ft_input_shape=None, feature_size=None, output_shape=None, hidden_units=1024, proj_units=128):
        # Pass parameters to the parent class
        super().__init__(base_encoder=base_encoder, input_shape=input_shape, feature_size=feature_size, output_shape=output_shape, hidden_units=hidden_units,proj_units=proj_units)
        self.ft_input_shape = ft_input_shape
        self.output_shape = output_shape

        # Additional FC layer before stacking all tiles back
        self.fc6 = nn.Sequential(
            nn.Linear(base_encoder.calc_feat_size(input_shape), 1024), #hard coded
            nn.ReLU())

        # Additional FC layer for jigsaw puzzle feature aggregation
        self.fc7 = nn.Sequential(
            nn.Linear(1024*9, 4096), #hard coded
            nn.ReLU())
        
        self.classifier_head = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Process each tile independently
        B, T, C, H, W = x.size()  # Batch, Tiles (9), Channels, Height, Width
        x = x.transpose(0, 1)
        x_list = []
        
        for i in range(9):
            # Get features from encoder for each tile
            z = self.encoder(x[i])
            # Project features if needed (base implementation would handle this)
            z = self.fc6(z) #like fc6
            z = z.view([B, 1, -1])
            x_list.append(z)
        
        # Concatenate features from all tiles
        x = torch.cat(x_list, 1)
        
        # Process aggregated features
        x = self.fc7(x.view(B, -1))
        x = self.classifier_head(x)
        
        return x