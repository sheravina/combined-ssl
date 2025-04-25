import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets
from utils.constants import *
from transformations import SimCLRTransformations
from transformations import basenorm_transformation, base_transformation, inet_transform, inet_simclr_transform

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 1:
        return transforms.functional.rotate(img, 90)
    elif rot == 2:
        return transforms.functional.rotate(img, 180)
    elif rot == 3:
        return transforms.functional.rotate(img, 270)
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DatasetWithRotation(torch.utils.data.Dataset):
    """Wrapper dataset that adds rotation augmentation to any dataset."""
    
    def __init__(self, base_dataset, seed=None) -> None:
        self.base_dataset = base_dataset
        # Create a separate random generator with the provided seed
        self.rng = random.Random(seed)
    
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, index: int):
        # Get original image and label
        image, cls_label = self.base_dataset[index]

        # Use the seeded random generator instead of global random
        rotation_label = self.rng.choice([0, 1, 2, 3])
        image_rotated = rotate_img(image, rotation_label)

        rotation_label = torch.tensor(rotation_label).long()
        # return image_rotated, rotation_label
        return image, image_rotated, rotation_label, torch.tensor(cls_label).long()

class DataManager:
    def __init__(self, dataset_name, ssl_method, batch_size, seed, encoder):
 
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.batch_size = batch_size
        self.seed = seed
        self.encoder = encoder

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

        if self.encoder == ENC_INET:
            self.transformation_train = inet_transform

            if self.ssl_method in [SSL_SIMCLR, SSL_SIMSIAM, SSL_VICREG]:
                self.transformation_test = inet_transform
                self.transformation_contrastive = SimCLRTransformations(
                    n_views=2,
                    include_original=True,
                    simclr_transform=inet_simclr_transform,
                    base_transform=inet_transform
                )

            elif self.ssl_method == SSL_ROTATION:
                self.transformation_test = inet_transform
                self.transformation_contrastive = inet_transform
            else:
                raise NotImplementedError("transformation not implemented yet")

        else:
            self.transformation_train = basenorm_transformation

            if self.ssl_method in [SSL_SIMCLR, SSL_SIMSIAM, SSL_VICREG]:
                self.transformation_test = basenorm_transformation
                self.transformation_contrastive = SimCLRTransformations(
                    n_views=2,
                    include_original=True
                )
            elif self.ssl_method == SSL_ROTATION:
                self.transformation_test = basenorm_transformation
                self.transformation_contrastive = basenorm_transformation
            else:
                raise NotImplementedError("transformation not implemented yet")

    
    
    def prepare_dataset(self):
        # train dataset for supervised model (transform to tensor + normalization)
        # contrastive dataset for unsupervised model and combined model (transform based on ssl method)
        # test dataset for test (transform to tensor only) -> does not work, so it has to be tensor + normalization
        if self.ssl_method == SSL_ROTATION:
            if self.dataset_name == DEBUG_DATASET:
                self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_train)
                base_contrastive = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_test)

            elif self.dataset_name == CIFAR10_DATASET:
                self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_train)
                base_contrastive = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.CIFAR10(root='./data_dir', train=False, download=True, transform=self.transformation_test)

            elif self.dataset_name == CIFAR100_DATASET:
                self.train_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True, transform=self.transformation_train)
                base_contrastive = datasets.CIFAR100(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=self.transformation_test)
                    
            elif self.dataset_name == IMAGENET_DATASET:
                self.train_dataset = datasets.ImageNet(root='./data_dir', split='train', download=True, transform=self.transformation_train)
                base_contrastive = datasets.ImageNet(root='./data_dir', split='train', download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.ImageNet(root='./data_dir', split='test', download=True, transform=self.transformation_test)

            elif self.dataset_name == CALTECH101_DATASET: # dataset has no embedded train and test splits
                self.train_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_train)
                base_contrastive = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.Caltech101(root='./data_dir', target_type='category', download=True, transform=self.transformation_test)

            else:
                raise NotImplementedError("dataset not implemented yet")
        else:

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
        # train_indices, test_indices, val_indices, sampled_indices = [], [], [], [] # sanity-checking indices
        if self.dataset_name == DEBUG_DATASET:
            
            # Collect all labels from the training dataset
            all_labels = []
            for i in range(len(self.train_dataset)):
                img, label = self.train_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)
            all_indices = np.arange(len(all_labels))

            # Take a stratified 10% sample first (if this is really needed)
            sampled_indices, _ = train_test_split(
                all_indices,
                test_size=0.9,  # Keep only 10% of the data
                random_state=self.seed,
                stratify=all_labels
            )

            # Get labels for the sampled subset
            sampled_labels = all_labels[sampled_indices]

            # Split the 10% sample into train (60%), val (20%), test (40%)
            test_indices, temp_indices = train_test_split(
                sampled_indices,
                test_size=0.6,  # 60% for temporary set
                random_state=self.seed,
                stratify=sampled_labels
            )
            
            # Get labels for the temp subset
            temp_labels = all_labels[temp_indices]
            
            # Second split: 20% validation, 20% test (from the 40% temp)
            train_indices, val_indices = train_test_split(
                temp_indices,
                test_size=0.1,  # Split temp 50-50 for val and test
                random_state=self.seed,
                stratify=temp_labels
            )

            # Create samplers with the generator
            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)
            val_sampler = SubsetRandomSampler(val_indices, generator=self.generator)
            test_sampler = SubsetRandomSampler(test_indices, generator=self.generator)
            
            # Create the data loaders
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=val_sampler)
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            valcont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=val_sampler)
            
            # Note: Using self.test_dataset here with indices from train_dataset 
            # only works if they have the same structure and ordering
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        elif self.dataset_name == CIFAR10_DATASET or CIFAR100_DATASET or IMAGENET_DATASET:
            all_labels = []
            for i in range(len(self.train_dataset)):
                img, label = self.train_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)

            # Take a stratified 10% sample first
            all_indices = np.arange(len(all_labels))
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=0.1,
                random_state=self.seed,
                stratify=all_labels
            )

            # Create samplers with the generator
            val_sampler = SubsetRandomSampler(val_indices, generator=self.generator)
            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)

            # train_loader and val_loader used for supervised methods, where we get from train dataset with normalize transforms
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # cont_loader and valcont_loader used for unsupervised task, where we get from train dataset with contrastive transforms
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            valcont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # test_loader for test_step() methods under supervised or fine-tuning 
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
            test_indices, temp_indices = train_test_split(
                all_indices,
                test_size=0.6,  # 40% for temporary set
                random_state=self.seed,
                stratify=all_labels
            )
            
            # Get labels for the temp subset
            temp_labels = all_labels[temp_indices]
            
            # Second split: 20% validation, 20% test (from the 40% temp)
            train_indices, val_indices = train_test_split(
                temp_indices,
                test_size=0.1,  # Split temp 50-50 for val and test
                random_state=self.seed,
                stratify=temp_labels
            )

            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)
            test_sampler = SubsetRandomSampler(test_indices, generator=self.generator)
            val_sampler = SubsetRandomSampler(val_indices, generator=self.generator)

        
            # train_loader and val_loader used for supervised methods, where we get from train dataset with normalize transforms
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader= DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # cont_loader and valcont_loader used for unsupervised task, where we get from train dataset with contrastive transforms
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            valcont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # test_loader for test_step() methods under supervised or fine-tuning, where we get from test dataset with normalize transforms          
            test_loader= DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        else:
            raise NotImplementedError("create_loader() for this dataset has not been implemented yet")
        
        # np.set_printoptions(threshold=10000, linewidth=200, edgeitems=10)

        # # Print lengths of each split
        # print(f"Train indices length: {len(train_indices)}")
        # print(f"Validation indices length: {len(val_indices)}")
        # print(f"Test indices length: {len(test_indices)}")

        # # Print actual indices (first 10 of each)
        # print("\nFirst 10 train indices:", train_indices)
        # print("\nFirst 10 validation indices:", val_indices)
        # print("\nFirst 10 test indices:", test_indices)

        # train_indices = list(train_indices)
        # val_indices = list(val_indices)
        # test_indices = list(test_indices)
        
        # # Verify no repeating indices
        # train_set = set(train_indices)
        # val_set = set(val_indices)
        # test_set = set(test_indices)
        
        # # Check and fix any overlaps (though train_test_split should prevent this)
        # train_val_overlap = train_set.intersection(val_set)
        # train_test_overlap = train_set.intersection(test_set)
        # val_test_overlap = val_set.intersection(test_set)
        
        # for idx in train_val_overlap:
        #     val_indices.remove(idx)
        
        # for idx in train_test_overlap:
        #     test_indices.remove(idx)
            
        # for idx in val_test_overlap:
        #     test_indices.remove(idx)
            
        # # Print verification messages
        # print(f"Final split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        # print(f"Total indices: {len(train_indices) + len(val_indices) + len(test_indices)}")
        # print(f"Original sampled size: {len(sampled_indices)}")
        
        return train_loader, cont_loader, test_loader, val_loader, valcont_loader


