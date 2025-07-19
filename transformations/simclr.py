from torchvision import transforms
# from transformations.constants_transformation import (
#     simclr_transformation,
#     train_transformation,
# )
from utils.constants import *


class SimCLRTransformations:
    def __init__(self, n_views=2, include_original=False, simclr_transform=None, base_transform=None, dataset_name=None):
        
        self.dataset_name = dataset_name
        self.create_transform_constants()

        # self.simclr_transforms = simclr_transform if simclr_transform is not None else simclr_transformation
        # self.base_transforms = base_transform if base_transform is not None else train_transformation
        self.n_views = n_views
        self.include_original = include_original



    def create_transform_constants(self):

        norm_mean = (0,0,0)
        norm_std = (0,0,0)

        if self.dataset_name in [DEBUG_DATASET, CIFAR10_DATASET, IMBV1_CIFAR10_DATASET, IMBV2_CIFAR10_DATASET,  IMBV3_CIFAR10_DATASET]:
            norm_mean = (0.4914, 0.4822, 0.4465)
            norm_std = (0.2023, 0.1994, 0.2010)
        elif self.dataset_name == CIFAR100_DATASET:
            norm_mean = (0.5071, 0.4865, 0.4409)
            norm_std = (0.2673, 0.2564, 0.2762)
        elif self.dataset_name == SVHN_DATASET:
            norm_mean =  (0.4377, 0.4438, 0.4728)
            norm_std = (0.1980, 0.2010, 0.1970)
        
        self.base_transformation = transforms.ToTensor()

        self.train_transformation = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ]
        )

        self.test_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ]
        )

        self.simclr_transformation = transforms.Compose(
            [   
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        self.inet_transform = transforms.Compose([
            transforms.Resize((299, 299)),  
            *self.train_transformation.transforms
        ])

        self.inet_simclr_transform = transforms.Compose([
            transforms.Resize((299, 299)), 
            transforms.RandomResizedCrop(size=299, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    def __call__(self, x):
        augmented_views = [self.simclr_transformation(x) for _ in range(self.n_views)]

        if self.include_original:
            original = self.train_transformation(x)
            augmented_views.append(original)

        return augmented_views

