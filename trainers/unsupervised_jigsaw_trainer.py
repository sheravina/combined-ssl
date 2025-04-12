import torch
from torch.autograd import Variable
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from .base_trainer import BaseTrainer

# Finetuner related packages
from models import UniversalFineTuner, SupervisedModel
import torch.nn as nn
import torch.optim as optim
from trainers import SupervisedTrainer

class JigsawTrainer(BaseTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, test_loader, ft_loader, optimizer, epochs):
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
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def train_step(self):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch, (images, labels) in enumerate(self.train_loader):
            permuted_tiles, permutation_idx, original_tiles, original_img = images[0].to(self.device), images[1].to(self.device),images[2].to(self.device), images[3].to(self.device)

            output = self.model(permuted_tiles)
            logits = Variable(output.argmax(dim=1).float(),requires_grad = True)
            permutation_idx = Variable(permutation_idx.float(),requires_grad = True)
            loss = self.criterion(logits, permutation_idx)
            train_loss += loss.item()

            correct += (logits == permutation_idx).sum().item()
            total += permutation_idx.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(self.train_loader)
        train_accuracy = correct / total
        return train_loss, train_accuracy
    
    def finetune_step(self):
        ft_model = UniversalFineTuner(self.model).to(self.device)
        # Only train the classifier layer
        ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(), lr=1e-4)
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
            ssl_loss, ssl_accuracy = self.train_step()
            results["ssl_loss"].append(ssl_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, SSL Loss: {ssl_loss:.4f}, SSL Accuracy: {ssl_accuracy:.4f}")

        ft_results = self.finetune_step().train()
        results.update(ft_results)
        
        return results