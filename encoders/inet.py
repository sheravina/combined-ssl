from torchvision.models import inception_v3
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class InceptionNet(BaseEncoder):
    """Inception‑V3 backbone encoder with classifier removed"""

    def __init__(self):
        super().__init__()
        model = inception_v3(weights=None, init_weights=True, aux_logits=False, transform_input=False)

        # Use full feature pipeline including avgpool
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            model.avgpool
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        if x.shape[-1] < 75 or x.shape[-2] < 75:
            raise ValueError("InceptionV3 requires input size ≥ 75x75 (preferably 299x299)")
        x = self.features(x)
        x = self.flatten(x)  # → (N, 2048)
        return x
