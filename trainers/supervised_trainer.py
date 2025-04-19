import torch
import os
import torch.nn as nn
from tqdm import tqdm
from losses.supervised_loss import SupervisedLoss
from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised learning."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs, save_dir=None):
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
        super().__init__(model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.save_dir = save_dir
        self.best_val_acc = 0

    def train_step(self, dataloader):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss, train_acc = 0, 0

        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            train_loss += loss.item() 
            
            pred_labels = pred.argmax(dim=1)
            train_acc += ((pred_labels == labels).sum().item()/len(pred_labels))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)  
        
        return train_loss, train_acc
    
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
                test_pred = self.model(images)
                loss = self.criterion(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
        
    def save_checkpoint(self, epoch, val_loss, train_loss, val_acc, train_acc):
        """
        Save model checkpoint - overwrites the previous best model
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            train_loss: Training loss
            val_acc: Validation accuracy
            train_acc: Training accuracy
        """
        # Only save if save_dir is provided
        if self.save_dir is None:
            return
            
        # Use a fixed filename for the best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'train_acc': train_acc
        }, best_model_path)
        
        print(f'New best model saved to {best_model_path} (Epoch {epoch+1}, Val Loss: {val_loss:.4f})')
                        
    def train(self):
        """
        Trains and tests the model for the specified number of epochs.

        Returns:
            results: Dictionary containing lists of training losses, test losses, and test accuracies
        """
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "eval_loss": [], "eval_acc": []}
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_step(self.train_loader)
            # Evaluate on test set
            val_loss, val_acc = self.test_step(self.val_loader)
            # Evaluate on test set
            eval_loss, eval_acc = self.test_step(self.test_loader)
            
            # Store results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["eval_loss"].append(eval_loss)
            results["eval_acc"].append(eval_acc)

            # Check if this is the best model so far based on validation loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_loss, train_loss, val_acc, train_acc)

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}")
        
        return results