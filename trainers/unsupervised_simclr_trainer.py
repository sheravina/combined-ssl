import torch
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from .base_trainer import BaseTrainer

# Finetuner related packages
from models import UniversalFineTuner, SupervisedModel
import torch.nn as nn
import torch.optim as optim
from trainers import SupervisedTrainer

class SimCLRTrainer(BaseTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, test_loader, ft_loader, optimizer, epochs,  temperature=0.5):
        """
        Initialize the SimCLR trainer.

        Args:
            model: The contrastive learning model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            device: Device to run training on ('cuda' or 'cpu')
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__(model, train_loader, test_loader, ft_loader, optimizer, epochs)
        self.temperature = temperature

    def train_step(self):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss = 0

        for batch, (images, labels) in enumerate(self.train_loader):
            #images, labels = images.to(self.device), labels.to(self.device)
            
            # calculate contrastive loss between two augmented images
            img_1, img_2 = images[0].to(self.device), images[1].to(self.device)
            b, c, h, w = images[0].shape
            input_ = torch.cat([img_1.unsqueeze(1), img_2.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            output = self.model(input_).view(b, 2, -1)

            loss = simclr_loss(output, self.temperature)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(self.train_loader)
        return train_loss
    
    def finetune_step(self):
        ft_model = UniversalFineTuner(self.model, num_classes=10).to(self.device)
    
        # Only train the classifier layer
        ft_optimizer = optim.Adam(ft_model.classifier.parameters(), lr=1e-4)

        ft_trainer = SupervisedTrainer(model=ft_model
                                            ,train_loader=self.ft_loader
                                            ,test_loader=self.test_loader
                                            ,ft_loader = None
                                            ,optimizer=ft_optimizer
                                            ,epochs=self.epochs)
        
        return ft_trainer

    
    def train(self):
        """
        Trains and tests the model for the specified number of epochs.

        Returns:
            results: Dictionary containing lists of training losses, test losses, and test accuracies
        """
        results = {"ssl_loss": []}
        
        for epoch in range(self.epochs):
            # Train for one epoch
            ssl_loss = self.train_step()
            
            # Store results
            results["ssl_loss"].append(ssl_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {ssl_loss:.4f}")

        
        ft_results = self.finetune_step().train()
        results.update(ft_results)
        
        return results