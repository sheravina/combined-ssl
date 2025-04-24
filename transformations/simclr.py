from torchvision import transforms
from transformations.constants_transformation import (
    simclr_transformation,
    basenorm_transformation,
)


class SimCLRTransformations:
    def __init__(self, n_views=2, include_original=False, simclr_transform=None, base_transform=None):
        self.simclr_transforms = simclr_transform if simclr_transform is not None else simclr_transformation
        self.base_transforms = base_transform if base_transform is not None else basenorm_transformation
        self.n_views = n_views
        self.include_original = include_original

    def __call__(self, x):
        augmented_views = [self.simclr_transforms(x) for _ in range(self.n_views)]

        if self.include_original:
            original = self.base_transforms(x)
            augmented_views.append(original)

        return augmented_views

