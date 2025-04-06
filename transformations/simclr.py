from torchvision import transforms
from transformations.constants_transformation import simclr_transformation, basenorm_transformation

class SimCLRTransformations:
    def __init__(self, n_views=2, include_original=False):
        self.simclr_transforms = simclr_transformation
        self.base_transforms = basenorm_transformation
        self.n_views = n_views
        self.include_original = include_original
        
    def __call__(self, x):
        augmented_views = [self.simclr_transforms(x) for i in range(self.n_views)]
        
        if self.include_original:
            # Apply minimal transforms to the original image if provided
            original = self.base_transforms(x)
                
            # Add original as the last view
            augmented_views.append(original)
            
        return augmented_views
