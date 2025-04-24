import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from encoders.base_encoder import BaseEncoder


class ViTEncoder(BaseEncoder):
    """Vision Transformer encoder using torchvision's ViT-B/16"""

    def __init__(self, img_size=224, in_channels=3, output_dim=None):
        super().__init__()

        # Load ViT without pretrained weights
        vit = vit_b_16(weights=None)

        # Replace patch embedding if input channels differ
        if in_channels != 3:
            vit.conv_proj = nn.Conv2d(
                in_channels,
                vit.conv_proj.out_channels,
                kernel_size=vit.conv_proj.kernel_size,
                stride=vit.conv_proj.stride,
                padding=vit.conv_proj.padding,
                bias=vit.conv_proj.bias is not None,
            )

        # Store components
        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token  # Use class_token (correct attribute)

        # Access pos_embedding correctly based on recent torchvision versions
        self.pos_embedding = vit.encoder.pos_embedding  # Updated path for positional embeddings

        # Access the transformer blocks in a different way
        self.blocks = vit.encoder.layers  # Using 'layers' instead of 'blocks'
        self.norm = vit.encoder.ln

        self.img_size = img_size
        self.patch_size = vit.patch_size
        self.embed_dim = vit.hidden_dim

        self.seq_length = (img_size // self.patch_size) ** 2 + 1

        # Optional projection head
        self.output_dim = output_dim
        if output_dim is not None:
            self.projection = nn.Linear(self.embed_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x):
        B = x.size(0)

        # Create patch embeddings
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add CLS token
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        # Resize positional embeddings if needed
        if x.size(1) != self.pos_embedding.size(1):
            pos_embed = self.interpolate_pos_embedding(x, self.pos_embedding, H, W)
        else:
            pos_embed = self.pos_embedding

        x = x + pos_embed

        # Transformer blocks (using layers)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract CLS token
        x = x[:, 0]

        # Project if needed
        if self.projection is not None:
            x = self.projection(x)

        return x

    def interpolate_pos_embedding(self, x, pos_embed, H, W):
        cls_pos = pos_embed[:, 0:1, :]
        patch_pos = pos_embed[:, 1:, :]

        num_patches = patch_pos.size(1)
        orig_size = int(num_patches ** 0.5)
        new_size = (H, W)

        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=new_size, mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat((cls_pos, patch_pos), dim=1)

    def calc_feat_size(self, input_shape):
        return self.output_dim if self.output_dim is not None else self.embed_dim
