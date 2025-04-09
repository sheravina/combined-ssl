import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel  # Import BaseModel from the base module

class CombinedSimCLR(BaseModel):
    def __init__(self,base_encoder,input_shape=None,feature_size=None,output_shape=None,hidden_units=1024,proj_units=128):
        # Pass parameters to the parent class
        super().__init__(base_encoder=base_encoder,input_shape=input_shape,feature_size=feature_size,output_shape=output_shape, hidden_units=hidden_units,proj_units=proj_units)

    def forward(self, x):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        # Get features from encoder
        h = self.encoder(x)
        # Get classification output and contrastive embeddings
        c = self.classifier_head(h)
        z = self.projection_head(h)
        z = F.normalize(z, p=2, dim=1)
        return c, z