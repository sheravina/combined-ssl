#VICReg model implementation from https://github.com/facebookresearch/vicreg/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from trainers import BaseUnsupervisedTrainer

# Finetuner related packages
from models import UniversalFineTuner, SupervisedModel
import torch.nn as nn
import torch.optim as optim
from trainers import SupervisedTrainer


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss

class VICRegTrainer(BaseUnsupervisedTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, save_dir=None):
        """
        Initialize the SimCLR trainer.

        Args:
            model: The contrastive learning model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            device: Device to run training on ('cuda' or 'cpu')
            temperature: Temperature parameter for contrastive loss
        """
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        super().__init__(model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, save_dir)

    def train_step(self, dataloader):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss = 0

        for batch, (images, labels) in enumerate(dataloader):
            #images, labels = images.to(self.device), labels.to(self.device)
            
            # calculate contrastive loss between two augmented images
            img_1, img_2 = images[0].to(self.device), images[1].to(self.device)

            z1, z2 = self.model(img_1, img_2)

            loss = (
            self.sim_coeff * invariance_loss(z1, z2)
            + self.std_coeff * variance_loss(z1, z2)
            + self.cov_coeff * covariance_loss(z1, z2)
            )

            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)  
            
        return train_loss
    
    # def finetune_step(self):
    #     ft_model = UniversalFineTuner(self.model).to(self.device)
    #     # Only train the classifier layer
    #     ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(), lr=1e-4)
    #     ft_trainer = SupervisedTrainer(model=ft_model
    #                                         ,train_loader=self.ft_loader
    #                                         ,test_loader=self.test_loader
    #                                         ,ft_loader = None
    #                                         ,optimizer=ft_optimizer
    #                                         ,epochs=self.epochs)
        
    #     return ft_trainer

    
    # def train(self):
    #     """
    #     Trains and tests the model for the specified number of epochs.

    #     Returns:
    #         results: Dictionary containing lists of training losses, test losses, and test accuracies
    #     """
    #     results = {"ssl_loss": []}
        
    #     for epoch in range(self.epochs):
    #         ssl_loss = self.train_step()
    #         results["ssl_loss"].append(ssl_loss)
    #         print(f"Epoch {epoch+1}/{self.epochs}, SSL Loss: {ssl_loss:.4f}")

    #     ft_results = self.finetune_step().train()
    #     results.update(ft_results)
        
    #     return results