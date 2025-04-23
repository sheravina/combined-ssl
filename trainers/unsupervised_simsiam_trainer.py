#simsiam model implementation from https://github.com/facebookresearch/simsiam
import torch
from tqdm import tqdm
from trainers import BaseUnsupervisedTrainer

# Finetuner related packages
from models import UniversalFineTuner, SupervisedModel
import torch.nn as nn
import torch.optim as optim
from trainers import SupervisedTrainer

class SimSiamTrainer(BaseUnsupervisedTrainer):
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
        super().__init__(model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, save_dir)
        self.criterion = nn.CosineSimilarity(dim=1).to(self.device)

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
            p1, p2, z1, z2 = self.model(x1=img_1, x2=img_2)
            loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
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