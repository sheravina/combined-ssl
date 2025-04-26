#VICReg model implementation from https://github.com/facebookresearch/vicreg/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .base_model import BaseModel  # Import BaseModel from the base module
from utils.constants import *

class CombinedVICReg(BaseModel):
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
        self.num_features = proj_units
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

        self.predictor_head = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer (projection)
        
    # original (facebook version) forward method with two images
    def forward(self, x, y, z):
        # Move tensor to the same device as model parameters
        x = self.projection_head(self.encoder(x))
        y = self.projection_head(self.encoder(y))
        z = self.encoder(z)
        pred = self.classifier_head(z)

        return x.detach(), y.detach(), pred