import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets
from utils.constants import *
from transformations import SimCLRTransformations, JigsawTransformations
from transformations import basenorm_transformation, base_transformation

class DataManager:
    def __init__(self, dataset_name, ssl_method, batch_size, seed):
 
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.batch_size = batch_size
        self.seed = seed

        # Set seeds for all random number generators
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.generator = torch.Generator().manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.create_contrastive_transform()
        self.prepare_dataset() 

    def create_contrastive_transform(self):
        self.transformation_train = basenorm_transformation   
          

        if self.ssl_method == SSL_SIMCLR:
            self.transformation_test = base_transformation  
            self.transformation_contrastive = SimCLRTransformations(n_views=2,include_original=True)
        elif self.ssl_method == SSL_JIGSAW:
            self.transformation_test = JigsawTransformations(num_permutations= 1000)
            self.transformation_contrastive = JigsawTransformations(num_permutations= 1000)
        else:
            raise NotImplementedError("transformation not implemented yet")
    
    def prepare_dataset(self):
        # train dataset for supervised model (transform to tensor + normalization)
        # contrastive dataset for unsupervised model and combined model (transform based on ssl method)
        # test dataset for test (transform to tensor only)

        if self.dataset_name == DEBUG_DATASET:
            self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_train)
            self.contrastive_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
            self.test_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_test) # with Train=True with test_sampler in the DataLoader()

        elif self.dataset_name == CIFAR10_DATASET:
            self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True,transform=self.transformation_train)
            self.contrastive_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True,transform=self.transformation_contrastive)
            self.test_dataset = datasets.CIFAR10(root='./data_dir', train=False, download=True, transform=self.transformation_test)

        elif self.dataset_name == CIFAR100_DATASET:
            self.train_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True,transform=self.transformation_train)
            self.contrastive_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True,transform=self.transformation_contrastive)
            self.test_dataset = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=self.transformation_test)
            
        elif self.dataset_name == IMAGENET_DATASET:
            self.train_dataset = datasets.ImageNet(root='./data_dir', split='train', download=True, transform=self.transformation_train)
            self.contrastive_dataset = datasets.ImageNet(root='./data_dir', split='train', download=True, transform=self.transformation_contrastive)
            self.test_dataset = datasets.ImageNet(root='./data_dir', split='test', download=True, transform=self.transformation_test)

        elif self.dataset_name == CALTECH101_DATASET: # dataset has no embedded train and test splits
            self.train_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_train)
            self.contrastive_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_contrastive)
            self.test_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_test)

        else:
            raise NotImplementedError("dataset not implemented yet")
        

    def create_loader(self):
        if self.dataset_name == DEBUG_DATASET:
            
            all_labels = []
            for i in range(len(self.train_dataset)):
                img, label = self.train_dataset[i]
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

            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)
            test_sampler = SubsetRandomSampler(test_indices, generator=self.generator)

            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader= DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        elif self.dataset_name == CIFAR10_DATASET or CIFAR100_DATASET or IMAGENET_DATASET:
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=self.generator)
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, shuffle=True, generator=self.generator)
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

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

            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)
            test_sampler = SubsetRandomSampler(test_indices, generator=self.generator)

            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader= DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        else:
            raise NotImplementedError("create_loader() for this dataset has not been implemented yet")

        
        return train_loader, cont_loader, test_loader


