from torchvision import transforms

class SimCLRTransformations:
    def __init__(self, n_views=2, include_original=False):
        self.simclr_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                    ])
        self.base_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                ])
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
