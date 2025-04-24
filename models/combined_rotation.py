import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel  # Import BaseModel from the base module

class CombinedRotation(BaseModel):
    def __init__(self,base_encoder,input_shape=None,feature_size=None,output_shape=None,output_shape_ssl=None, output_shape_sup=None,hidden_units=1024,proj_units=128):
        # Pass parameters to the parent class
        super().__init__(base_encoder=base_encoder,input_shape=input_shape,feature_size=feature_size,output_shape=output_shape, hidden_units=hidden_units,proj_units=proj_units)
        
        self.output_shape_ssl = output_shape_ssl
        self.output_shape_sup = output_shape_sup

        self.classifier_head_ssl = nn.Linear(in_features=self.feature_size, out_features=self.output_shape_ssl)
        self.classifier_head_sup = nn.Linear(in_features=self.feature_size, out_features=self.output_shape_sup)

    def forward(self, x1, x2):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x1 = x1.to(device)
        x2 = x2.to(device)
        # Get features from encoder
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        # Get classification output and contrastive embeddings
        c1 = self.classifier_head_ssl(h1)
        c2 = self.classifier_head_sup(h2)

        return c1, c2