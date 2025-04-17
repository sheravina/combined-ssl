import os
import torch
from tqdm import tqdm


class BaseCombinedTrainer:
    """Base class for all trainers."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, optimizer, lr_scheduler, epochs, save_dir=None):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            device: Device to run training on ('cuda' or 'cpu')
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.ft_loader = ft_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.save_dir = save_dir

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
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "eval_loss": [], "eval_acc": [], "ssl_loss": [], "sup_loss": []}
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc, ssl_loss, sup_loss = self.train_step(self.train_loader)
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
            results["ssl_loss"].append(ssl_loss)
            results["sup_loss"].append(sup_loss)

            # Check if this is the best model so far based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, train_loss, val_acc, train_acc)

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")
        
        return results
    