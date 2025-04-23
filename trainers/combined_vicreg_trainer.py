# adapted from simsiam model implementation from https://github.com/facebookresearch/simsiam
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from losses.supervised_loss import SupervisedLoss
from trainers import BaseCombinedTrainer


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

class CombinedVICRegTrainer(BaseCombinedTrainer):
    """Trainer for combined supervised and contrastive learning."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs, alpha=0.5, save_dir=None):
        """
        Initialize the combined trainer.

        Args:
            model: The combined model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            device: Device to run training on ('cuda' or 'cpu')
            temperature: Temperature parameter for contrastive loss
            alpha: Weight for supervised loss
        """
        super().__init__(model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs, save_dir)
        self.alpha = alpha
        self.criterion = SupervisedLoss()
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def train_step(self, dataloader):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss, train_acc = 0, 0
        ssl_loss, sup_loss = 0, 0

        for batch, (images, labels) in enumerate(dataloader):
            img_1, img_2, img_3, labels = images[0].to(self.device), images[1].to(self.device), images[2].to(self.device), labels.to(self.device)

            z1, z2, pred = self.model(img_1, img_2, img_3)
            
            contrastive_loss = (
            self.sim_coeff * invariance_loss(z1, z2)
            + self.std_coeff * variance_loss(z1, z2)
            + self.cov_coeff * covariance_loss(z1, z2)
            )

            ssl_loss += contrastive_loss.item()

            supervised_loss = self.criterion(pred, labels)
            sup_loss += supervised_loss.item()

            # Combined loss
            loss = contrastive_loss + self.alpha * supervised_loss
            train_loss += loss.item()
            pred_labels = pred.argmax(dim=1)
            train_acc += ((pred_labels == labels).sum().item()/len(pred_labels))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        ssl_loss = ssl_loss / len(dataloader)
        sup_loss = sup_loss / len(dataloader)
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)

        return train_loss, train_acc, ssl_loss, sup_loss
    
    def test_step(self, dataloader):
        """
        Tests the model for a single epoch

        Returns:
            test_loss, test_acc
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                _, _, test_pred = self.model(images, images, images)
                loss = self.criterion(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc