import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
from torchvision import datasets
from utils.constants import *
from transformations import SimCLRTransformations
# from transformations import train_transformation, test_transformation, inet_transform, inet_simclr_transform

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
        
        print(f"CUDA is available:{torch.cuda.is_available()}")

        self.create_transform_constants()
        self.create_contrastive_transform()
        self.prepare_dataset() 

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

    def create_contrastive_transform(self):

        if self.encoder == ENC_INET:
            self.transformation_train = self.inet_transform

            if self.ssl_method in [SSL_SIMCLR, SSL_SIMSIAM, SSL_VICREG]:
                self.transformation_test = self.inet_transform
                self.transformation_contrastive = SimCLRTransformations(
                    n_views=2,
                    include_original=True,
                    simclr_transform=self.inet_simclr_transform,
                    base_transform=self.inet_transform,
                    dataset_name=self.dataset_name
                )

            elif self.ssl_method == SSL_ROTATION:
                self.transformation_test = self.inet_transform
                self.transformation_contrastive = self.inet_transform
            else:
                raise NotImplementedError("transformation not implemented yet")

        else:
            self.transformation_train = self.train_transformation

            if self.ssl_method in [SSL_SIMCLR, SSL_SIMSIAM, SSL_VICREG]:
                self.transformation_test = self.test_transformation
                self.transformation_contrastive = SimCLRTransformations(
                    n_views=2,
                    include_original=True,
                    dataset_name=self.dataset_name
                )
            elif self.ssl_method == SSL_ROTATION:
                self.transformation_test = self.test_transformation
                self.transformation_contrastive = self.train_transformation
            else:
                raise NotImplementedError("transformation not implemented yet")

    def _generate_all_labels_batches(self, labels_for_sampler, batch_size):
        """
        A generator that yields batches of indices, ensuring each batch
        contains at least one sample from every unique label.
        """
        # Get the indices relative to the subset we are sampling from (0 to N-1)
        sampler_indices = np.arange(len(labels_for_sampler))
        unique_labels = np.unique(labels_for_sampler)
        
        # A map from class label to the indices of samples with that label
        class_indices_map = {lbl: sampler_indices[labels_for_sampler == lbl] for lbl in unique_labels}
        
        if batch_size < len(unique_labels):
            raise ValueError(
                f"Batch size ({batch_size}) must be greater than or equal to the "
                f"number of unique classes ({len(unique_labels)}) to ensure all labels are in a batch."
            )
            
        available_indices = set(sampler_indices)
        
        # Use the class's own seeded random number generator
        rng = self.rng if hasattr(self, 'rng') else np.random.RandomState(self.seed)

        while len(available_indices) > 0:
            batch = []
            # 1. Add one random, available sample from each class
            for lbl in unique_labels:
                available_class_indices = list(set(class_indices_map[lbl]) & available_indices)
                if not available_class_indices:
                    continue
                    
                chosen_index = rng.choice(available_class_indices)
                batch.append(chosen_index)
                available_indices.remove(chosen_index)
                
            # 2. Fill the rest of the batch
            remaining_batch_size = batch_size - len(batch)
            if remaining_batch_size > 0 and len(available_indices) > 0:
                
                # --- THIS IS THE CORRECTED LINE ---
                fill_indices = rng.choice(
                    list(available_indices),
                    size=min(remaining_batch_size, len(available_indices)),
                    replace=False
                )
                batch.extend(fill_indices)
                available_indices.difference_update(fill_indices)
                
            rng.shuffle(batch)
            yield batch
    
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

            elif self.dataset_name in [CIFAR10_DATASET, IMBV1_CIFAR10_DATASET, IMBV2_CIFAR10_DATASET,  IMBV3_CIFAR10_DATASET]:
                self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_train)
                base_contrastive = datasets.CIFAR10(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.CIFAR10(root='./data_dir', train=False, download=True, transform=self.transformation_test)

            elif self.dataset_name == CIFAR100_DATASET:
                self.train_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True, transform=self.transformation_train)
                base_contrastive = datasets.CIFAR100(root='./data_dir', train=True, download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=self.transformation_test)

            elif self.dataset_name == SVHN_DATASET:
                self.train_dataset = datasets.SVHN(root='./data_dir', split="train", download=True, transform=self.transformation_train)
                base_contrastive = datasets.SVHN(root='./data_dir', split="train", download=True, transform=self.transformation_contrastive)
                self.contrastive_dataset = DatasetWithRotation(base_contrastive, seed=self.seed)
                self.test_dataset = datasets.SVHN(root='./data_dir', split="test", download=True, transform=self.transformation_test)

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

            elif self.dataset_name in [CIFAR10_DATASET, IMBV1_CIFAR10_DATASET, IMBV2_CIFAR10_DATASET,  IMBV3_CIFAR10_DATASET]:
                self.train_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True,transform=self.transformation_train)
                self.contrastive_dataset = datasets.CIFAR10(root='./data_dir', train=True, download=True,transform=self.transformation_contrastive)
                self.test_dataset = datasets.CIFAR10(root='./data_dir', train=False, download=True, transform=self.transformation_test)

            elif self.dataset_name == CIFAR100_DATASET:
                self.train_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True,transform=self.transformation_train)
                self.contrastive_dataset = datasets.CIFAR100(root='./data_dir', train=True, download=True,transform=self.transformation_contrastive)
                self.test_dataset = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=self.transformation_test)

            elif self.dataset_name == SVHN_DATASET:
                self.train_dataset = datasets.SVHN(root='./data_dir', split="train", download=True,transform=self.transformation_train)
                self.contrastive_dataset = datasets.SVHN(root='./data_dir', split="train", download=True,transform=self.transformation_contrastive)
                self.test_dataset = datasets.SVHN(root='./data_dir', split="test", download=True, transform=self.transformation_test)            
                            
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

        elif self.dataset_name in [CIFAR10_DATASET, CIFAR100_DATASET, IMAGENET_DATASET, SVHN_DATASET]:
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

        elif self.dataset_name == IMBV1_CIFAR10_DATASET:
            if hasattr(self.train_dataset, 'targets'):
                original_all_labels = np.array(self.train_dataset.targets)
            else:
                original_all_labels = np.array([label for _, label in self.train_dataset])

            original_all_indices = np.arange(len(original_all_labels))

            imbalanced_indices = []

            rng = np.random.RandomState(self.seed)

            for class_idx in range(10):
                class_indices = original_all_indices[original_all_labels == class_idx]
                if class_idx < 5:
                    imbalanced_indices.extend(class_indices)
                else:
                    num_samples_to_keep = int(0.1 * len(class_indices))
                    if num_samples_to_keep == 0 and len(class_indices) > 0:
                        num_samples_to_keep = 1

                    reduced_indices = rng.choice(
                        class_indices,
                        size=num_samples_to_keep,
                        replace=False
                    )
                    imbalanced_indices.extend(reduced_indices)

            imbalanced_indices = np.array(imbalanced_indices)
            rng.shuffle(imbalanced_indices)

            imbalanced_dataset = Subset(self.train_dataset, imbalanced_indices)
            imbalanced_labels = original_all_labels[imbalanced_indices]
            
            # We still split to get the indices for train and validation
            train_relative_indices, val_relative_indices = train_test_split(
                np.arange(len(imbalanced_dataset)),
                test_size=0.1,
                random_state=self.seed,
                stratify=imbalanced_labels
            )
            
            # --- START OF CHANGES ---

            # 1. Get the labels for the training subset to pass to our generator
            train_labels_for_sampler = imbalanced_labels[train_relative_indices]

            # 2. Create the train and validation subsets that the DataLoaders will use
            train_subset = Subset(imbalanced_dataset, train_relative_indices)
            val_subset = Subset(imbalanced_dataset, val_relative_indices)
            
            # Also create subsets for the contrastive dataset
            contrastive_train_subset = Subset(self.contrastive_dataset, imbalanced_indices[train_relative_indices])
            contrastive_val_subset = Subset(self.contrastive_dataset, imbalanced_indices[val_relative_indices])

            # 3. Create the batch sampler by calling our new generator method
            # This returns a generator object that yields lists of indices.
            train_batch_sampler = list(self._generate_all_labels_batches(train_labels_for_sampler, self.batch_size))

            # 4. Create DataLoaders using `batch_sampler`. 
            # NOTE: When using `batch_sampler`, you MUST NOT use `batch_size`, `shuffle`, or `sampler`.
            train_loader = DataLoader(dataset=train_subset, batch_sampler=train_batch_sampler)
            cont_loader = DataLoader(dataset=contrastive_train_subset, batch_sampler=train_batch_sampler)

            # Validation loader can remain the same, as balanced batches are less critical here.
            val_sampler = SubsetRandomSampler(np.arange(len(val_subset)), generator=self.generator)
            val_loader = DataLoader(dataset=val_subset, batch_size=self.batch_size, sampler=val_sampler)
            valcont_loader = DataLoader(dataset=contrastive_val_subset, batch_size=self.batch_size, sampler=val_sampler)

            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # --- END OF CHANGES ---

        elif self.dataset_name == IMBV2_CIFAR10_DATASET:
            all_labels = []
            for i in range(len(self.train_dataset)):
                img, label = self.train_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)
            all_indices = np.arange(len(all_labels))

            # Create imbalance by reducing samples from some classes
            imbalanced_indices = []
            
            # Use NumPy's RandomState with your seed for consistent results
            rng = np.random.RandomState(self.seed)
            
            for class_idx in range(10):  # CIFAR-10 has 10 classes
                class_indices = all_indices[all_labels == class_idx]
                if class_idx % 2 == 0:  # Keep all samples for first 5 classes
                    imbalanced_indices.extend(class_indices)
                else:  # Keep only 10% for last 5 classes
                    # Use the seeded random number generator
                    reduced_indices = rng.choice(
                        class_indices, 
                        size=int(0.1 * len(class_indices)), 
                        replace=False
                    )
                    imbalanced_indices.extend(reduced_indices)
            
            # Use the imbalanced set of indices
            all_indices = np.array(imbalanced_indices)
            all_labels = all_labels[all_indices]
            
            # Split into train and validation without stratification
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=0.1,
                random_state=self.seed,
                stratify=None  # No stratification for imbalanced dataset
            )

            # Create samplers with the generator
            val_sampler = SubsetRandomSampler(val_indices, generator=self.generator)
            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)

            # train_loader and val_loader used for supervised methods
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # cont_loader and valcont_loader used for unsupervised task
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            valcont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # test_loader for test_step() methods
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False) 

        elif self.dataset_name == IMBV3_CIFAR10_DATASET:
            all_labels = []
            for i in range(len(self.train_dataset)):
                img, label = self.train_dataset[i]
                all_labels.append(label)

            # Convert to numpy array for easier handling
            all_labels = np.array(all_labels)
            all_indices = np.arange(len(all_labels))

            # Create imbalance by reducing samples from some classes
            imbalanced_indices = []
            
            # Use NumPy's RandomState with your seed for consistent results
            rng = np.random.RandomState(self.seed)
            
            for class_idx in range(10):  # CIFAR-10 has 10 classes
                class_indices = all_indices[all_labels == class_idx]
                if class_idx in [0, 1, 9, 8, 5]:  # Keep all samples for first 5 classes
                    imbalanced_indices.extend(class_indices)
                else:  # Keep only 10% for last 5 classes
                    # Use the seeded random number generator
                    reduced_indices = rng.choice(
                        class_indices, 
                        size=int(0.1 * len(class_indices)), 
                        replace=False
                    )
                    imbalanced_indices.extend(reduced_indices)
            
            # Use the imbalanced set of indices
            all_indices = np.array(imbalanced_indices)
            all_labels = all_labels[all_indices]
            
            # Split into train and validation without stratification
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=0.1,
                random_state=self.seed,
                stratify=None  # No stratification for imbalanced dataset
            )

            # Create samplers with the generator
            val_sampler = SubsetRandomSampler(val_indices, generator=self.generator)
            train_sampler = SubsetRandomSampler(train_indices, generator=self.generator)

            # train_loader and val_loader used for supervised methods
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # cont_loader and valcont_loader used for unsupervised task
            cont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=train_sampler)
            valcont_loader = DataLoader(dataset=self.contrastive_dataset, batch_size=self.batch_size, sampler=val_sampler)

            # test_loader for test_step() methods
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False) 

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

        # print(f"Original sampled size: {len(sampled_indices)}")
        
        return train_loader, cont_loader, test_loader, val_loader, valcont_loader


