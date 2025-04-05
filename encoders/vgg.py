import torch.nn as nn
from encoders.base_encoder import BaseEncoder


class VGGEncoder(BaseEncoder):
    def __init__(self, in_channels=3, hidden_units=10):
        super().__init__()

        # Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return x
