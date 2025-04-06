import torch
import torch.nn as nn
from tqdm import tqdm
from losses.supervised_loss import SupervisedLoss
from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised learning."""

    def __init__(self, model, train_loader, test_loader, ft_loader, optimizer, epochs):
        """
        Initialize the supervised trainer.

        Args:
            model: The supervised model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            epochs: Number of epochs to train for
            device: Device to run training on (if None, will use 'cuda' if available, else 'cpu')
        """
        # Auto-detect device if not specified
        super().__init__(model, train_loader, test_loader, ft_loader, optimizer, epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss = 0

        for batch, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            train_loss += loss.item() 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(self.train_loader)
        return train_loss
    
    def test_step(self):
        """
        Tests the model for a single epoch

        Returns:
            test_loss, test_acc
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                test_pred = self.model(images)
                loss = self.criterion(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(self.test_loader)
        test_acc = test_acc / len(self.test_loader)
        return test_loss, test_acc
        
            
    def train(self):
        """
        Trains and tests the model for the specified number of epochs.

        Returns:
            results: Dictionary containing lists of training losses, test losses, and test accuracies
        """
        results = {"train_loss": [], "test_loss": [], "test_acc": []}
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self.train_step()
            # Evaluate on test set
            test_loss, test_acc = self.test_step()
            
            # Store results
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return results