import torch
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class ConvNetEncoder(BaseEncoder):
    """
    A 13-Layer MaxPooling ConvNet with BatchNormalization, translated from Keras.
    https://github.com/vuptran/sesemi/blob/master/networks/convnet.py

    This network architecture is based on the provided Keras model, maintaining the
    same structure of convolutional layers, batch normalization, leaky ReLU activations,
    and max pooling.
    """

    def __init__(self, in_channels=3, dropout=0.0):
        """
        Initializes the 13-layer ConvNet.

        Args:
            in_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            dropout (float): Dropout probability. If 0.0, no dropout is applied.
        """
        super().__init__()

        leakiness = 0.1
        
        # Note on Weight Decay (L2 Regularization):
        # In PyTorch, weight decay is typically handled by the optimizer (e.g., AdamW
        # or by passing the `weight_decay` parameter to an optimizer like SGD or Adam),
        # not within the layer definitions as in Keras.
        # Example: optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

        # --- Block 1: 3x Conv128 -> MaxPool ---
        b1 = 128
        b2 = 256
        b3 = 512

        block1_layers = [
            nn.Conv2d(in_channels, b1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b1, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b1, b1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b1, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b1, b1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b1, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        if dropout > 0.0:
            block1_layers.append(nn.Dropout(dropout))
        self.block_1 = nn.Sequential(*block1_layers)

        # --- Block 2: 3x Conv256 -> MaxPool ---
        block2_layers = [
            nn.Conv2d(b1, b2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b2, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b2, b2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b2, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b2, b2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(b2, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        if dropout > 0.0:
            block2_layers.append(nn.Dropout(dropout))
        self.block_2 = nn.Sequential(*block2_layers)
        
        # --- Block 3: Final Convolutional Layers ---
        # Keras 'valid' padding corresponds to padding=0 in PyTorch.
        self.block_3 = nn.Sequential(
            nn.Conv2d(b2, b3, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(b3, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b3, b2, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(b2, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),

            nn.Conv2d(b2, b1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(b1, momentum=0.99, eps=0.001),
            nn.LeakyReLU(leakiness, inplace=True),
        )

        # --- Flatten Layer ---
        self.flatten = nn.Flatten()
        
        # --- Initialize Weights ---
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the convolutional layers using Kaiming Normal
        (He Normal) initialization, which is the PyTorch equivalent of
        Keras' `initializers.he_normal()`.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The flattened feature tensor.
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.flatten(x)
        return x