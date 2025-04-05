# select a dataset (debug, cifar10, cifar100, caltech101, imagenet)
#pseudocode

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils.constants import *
from transformations import SimCLRTransformations

class DataManager:
    def __init__(self, dataset_name, ssl_method, batch_size):
 
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.batch_size = batch_size

        self.create_contrastive_transform()
        self.prepare_dataset() 

    def create_contrastive_transform(self):
        if self.ssl_method == SSL_SIMCLR:
            self.transformation = SimCLRTransformations(n_views=2,include_original=True)
        else:
            raise NotImplementedError("transformation not implemented yet")
    
    def prepare_dataset(self):
        if self.dataset_name == DEBUG_DATASET:
            self.contrastive_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation)
            self.test_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=ToTensor()) # with Train=True with test_sampler in the DataLoader()

        elif self.dataset_name == CIFAR10_DATASET:
            self.contrastive_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True,transform=self.transformation)
            self.test_dataset = datasets.CIFAR10(root='./data_dir', train=False, download=True, transform=ToTensor())

        elif self.dataset_name == CIFAR100_DATASET:
            self.contrastive_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True,transform=self.transformation)
            self.test_dataset = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=ToTensor())
            
        elif self.dataset_name == IMAGENET_DATASET:
            self.contrastive_dataset = datasets.ImageNet(root='./data_dir', split='train', download=True, transform=self.transformation)
            self.test_dataset = datasets.ImageNet(root='./data_dir', split='test', download=True, transform=ToTensor())

        elif self.dataset_name == CALTECH101_DATASET: # dataset has no embedded train and test splits
            self.contrastive_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation)
            self.test_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=ToTensor())

        else:
            raise NotImplementedError("dataset not implemented yet")
        

    def create_loader(self):
        if self.dataset_name == DEBUG_DATASET:
            
            all_labels = []
            for i in range(len(self.test_dataset)):
                img, label = self.test_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)

            # Take a stratified 10% sample first
            all_indices = np.arange(len(all_labels))
            sampled_indices, _ = train_test_split(
                all_indices,
                test_size=0.9,  # Keep only 10% of the data
                random_state=42,
                stratify=all_labels
            )

            # Get labels for the sampled subset
            sampled_labels = all_labels[sampled_indices]

            # Now split the 10% sample into train, val, test sets
            # First split: 60% train, 40% test
            train_indices, test_indices = train_test_split(
                np.arange(len(sampled_indices)),
                test_size=0.4,
                random_state=42,
                stratify=sampled_labels
            )

            # Convert relative indices to absolute indices
            train_indices = [sampled_indices[i] for i in train_indices]
            test_indices = [sampled_indices[i] for i in test_indices]

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader= DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        elif self.dataset_name == CIFAR10_DATASET or CIFAR100_DATASET or IMAGENET_DATASET:
            train_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size)
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)

        elif self.dataset_name == CALTECH101_DATASET:
            all_labels = []
            for i in range(len(self.test_dataset)):
                img, label = self.test_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)

            # Take a stratified 10% sample first
            all_indices = np.arange(len(all_labels))
            train_indices, test_indices = train_test_split(
                all_indices,
                test_size=0.4,  # Keep only 10% of the data
                random_state=42,
                stratify=all_labels
            )

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader= DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        else:
            raise NotImplementedError("create_loader() for this dataset has not been implemented yet")

        
        return train_loader, test_loader


