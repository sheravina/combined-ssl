#simsiam model implementation from https://github.com/facebookresearch/simsiam
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel  # Import BaseModel from the base module
from utils.constants import *

class SimSiam(BaseModel):
    def __init__(self, base_encoder, input_shape=None, feature_size=None, output_shape=None, hidden_units=512, proj_units=2048):
        # Pass parameters to the parent class
        super().__init__(base_encoder=base_encoder, input_shape=input_shape, feature_size=feature_size, output_shape=output_shape, hidden_units=hidden_units,proj_units=proj_units)
        self.output_shape = output_shape
        self.ft_input_shape = input_shape

        if self.encoder == ENC_RESNET18 or self.encoder == ENC_RESNET50:
            self.encoder = base_encoder(zero_init_residual=True)
        else:
            self.encoder = base_encoder
        
        prev_dim = base_encoder.calc_feat_size(input_shape)
        dim = proj_units
        pred_dim = hidden_units
        self.projection_final = nn.Linear(prev_dim, dim)
        self.projection_head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.projection_final,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projection_head[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.predictor_head = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer (projection)
        
    # original (facebook version) forward method with two images
    def forward(self, x1, x2):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x1, x2 = x1.to(device), x2.to(device)
        # Get features from encoder
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        # Project features to the contrastive embedding space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        # Pass to a predictor 
        p1 = self.predictor_head(z1)
        p2 = self.predictor_head(z2)
        return p1, p2, z1.detach(), z2.detach()
    
    # forward method with single image
    # def forward(self, x):
    #     # Move tensor to the same device as model parameters
    #     device = next(self.parameters()).device
    #     x = x.to(device)
    #     # Get features from encoder
    #     h = self.encoder(x)
    #     # Project features to the contrastive embedding space
    #     z = self.projection_head(h)
    #     # Pass to a predictor 
    #     p = self.predictor_head(z)
    #     return p, z