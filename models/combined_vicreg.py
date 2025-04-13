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
        
        self.batch_multiplier = 4
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

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
        x = self.predictor_head(self.projection_head(self.encoder(x)))
        y = self.predictor_head(self.projection_head(self.encoder(y)))
        z = self.encoder(z)
        pred = self.classifier_head(z)

        repr_loss = F.mse_loss(x, y)

        # x = torch.cat(SimulatedGatherLayer.apply(x), dim=0) #distributed training not used
        # y = torch.cat(SimulatedGatherLayer.apply(y), dim=0) #distributed training not used
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        batch_size = x.size(0)
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss, pred
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
# class SimulatedGatherLayer(torch.autograd.Function):
#     """
#     A layer that simulates the gathering effect of distributed training
#     by creating multiple copies of the batch with slight perturbations.
#     This effectively increases the batch size for variance and covariance calculations.
    
#     Importantly, it returns a TUPLE of tensors, just like the original FullGatherLayer would.
#     """

#     @staticmethod
#     def forward(ctx, x, num_copies=4):
#         batch_size = x.size(0)
#         result = []
        
#         # First element is the original batch
#         result.append(x)
        
#         # Create additional copies with small perturbations
#         for i in range(num_copies - 1):
#             # Add small noise to simulate different batches from different processes
#             noise_scale = 0.001
#             noise = torch.randn_like(x) * noise_scale
#             result.append(x + noise)
        
#         # Save context for backward
#         ctx.batch_size = batch_size
#         ctx.num_copies = num_copies
        
#         # Return a TUPLE of tensors
#         return tuple(result)

#     @staticmethod
#     def backward(ctx, *grads):
#         # Sum all gradients to simulate all_reduce
#         combined_grad = torch.zeros_like(grads[0])
#         for g in grads:
#             combined_grad += g
            
#         return combined_grad


# class FullGatherLayer(torch.autograd.Function):
#     """
#     Gather tensors from all process and support backward propagation
#     for the gradients across processes.
#     """

#     @staticmethod
#     def forward(ctx, x):
#         output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
#         dist.all_gather(output, x)
#         return tuple(output)

#     @staticmethod
#     def backward(ctx, *grads):
#         all_gradients = torch.stack(grads)
#         dist.all_reduce(all_gradients)
#         return all_gradients[dist.get_rank()]