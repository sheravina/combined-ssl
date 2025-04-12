# adapted from https://github.com/bbrattoli/JigsawPuzzlePytorch
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import random
from transformations.constants_transformation import (
    basenorm_jp_transformation,
    jigsaw_transformation,
)

class JigsawTransformations(data.Dataset):
    def __init__(self, num_permutations=1000): #, seed=None|
        self.num_permutations = num_permutations 
        self.permutations = self.generate_permutations()
        # Original image transformer
        self.original_transformer = basenorm_jp_transformation        
        # Tile augmentation
        self.augment_tile = jigsaw_transformation
    
    def generate_permutations(self):
        """Generate permutations ensuring sufficient distance from identity."""
        permutations = []
        permutations.append(list(range(9)))  # Add identity permutation
        
        def is_valid_permutation(perm):
            # Count how many tiles are in their original position
            in_place = sum(i == p for i, p in enumerate(perm))
            # We want at least 4 tiles to be moved
            return in_place <= 5
        
        while len(permutations) < self.num_permutations:
            perm = list(range(9))
            random.shuffle(perm)
            if is_valid_permutation(perm) and perm not in permutations:
                permutations.append(perm)
        
        return np.array(permutations)
        
    def __call__(self, img):           
        # Store original normalized image
        img = self.original_transformer(img)
        
        # Create 3x3 grid of tiles directly from tensor
        # CIFAR-10 images are 32x32
        tiles = []
        for i in range(3):
            for j in range(3):
                # Extract tile (roughly 10x10)
                tile = img[:, i*10:min((i+1)*10, 32), j*10:min((j+1)*10, 32)]
                # Apply augmentation
                tile = self.augment_tile(tile)
                tiles.append(tile)
        
        # Select random permutation
        order = np.random.randint(len(self.permutations))
        permutation = self.permutations[order]
        
        # Apply permutation to tiles
        permuted_tiles = [tiles[p] for p in permutation]
        permuted_tiles = torch.stack(permuted_tiles)
        original_tiles = torch.stack(tiles)

        augmented_views = []
        augmented_views.append(permuted_tiles)
        augmented_views.append(int(order))
        augmented_views.append(original_tiles)
        augmented_views.append(img)
        
        # Return permuted tiles, permutation index, original tiles, original image, class label
        return augmented_views