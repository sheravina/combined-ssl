import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import UniversalFineTuner, SupervisedModel
from trainers import SupervisedTrainer
from utils.constants import *
from nntrain import LARS


class BaseUnsupervisedTrainer:
    """Base class for all trainers."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, save_dir=None):
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
        self.optimizer_name = optimizer_name
        self.lr_scheduler = lr_scheduler
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.epochs_pt = epochs_pt
        self.epochs_ft = epochs_ft
        self.save_dir = save_dir

        self.valcont_loader = valcont_loader
        self.best_val_loss = float('inf')


    def finetune_step(self):
        # Load the best model if it exists
        best_model_path = os.path.join(self.save_dir, 'best_ssl_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best SSL model from epoch {checkpoint['epoch']+1} for finetuning")
        else:
            print("No best model found, using current model for finetuning")
        
        ft_model = UniversalFineTuner(self.model).to(self.device)
        # Only train the classifier layer

        if self.optimizer_name == OPT_ADAM:
            ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == OPT_LARS:
            ft_optimizer = LARS(ft_model.classifier_head.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

        elif self.optimizer_name == OPT_SGD:
            ft_optimizer = optim.SGD(ft_model.classifier_head.parameters(),lr=self.learning_rate,momentum=0.9, weight_decay=self.weight_decay, nesterov=False)

        # ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(), lr=1e-4)


        ft_trainer = SupervisedTrainer(model=ft_model
                                            ,train_loader=self.ft_loader
                                            ,test_loader=self.test_loader
                                            ,val_loader=self.val_loader
                                            ,ft_loader = None
                                            ,optimizer=ft_optimizer
                                            ,lr_scheduler=self.lr_scheduler
                                            ,epochs=self.epochs_ft
                                            ,save_dir=self.save_dir)
        
        return ft_trainer
    
    def save_checkpoint(self, epoch, ssl_train_loss, ssl_val_loss):
        """
        Save model checkpoint - overwrites the previous best model
        
        Args:
            epoch: Current epoch
            ssl_loss: SSL training loss
            ft_val_loss: Finetuning validation loss (optional)
            ft_train_loss: Finetuning training loss (optional)
            ft_val_acc: Finetuning validation accuracy (optional)
            ft_train_acc: Finetuning training accuracy (optional)
        """            
        # Use a fixed filename for the best model
        best_model_path = os.path.join(self.save_dir, 'best_ssl_model.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'ssl_train_loss' : ssl_train_loss,
            'ssl_val_loss': ssl_val_loss
        }
            
        torch.save(checkpoint, best_model_path)
        
        print(f'New best SimCLR model saved to {best_model_path} (Epoch {epoch+1}, SSL Train Loss: {ssl_train_loss:.4f}, SSL Val Loss: {ssl_val_loss:.4f})')

    
    def train(self):
        """
        Trains and tests the model for the specified number of epochs.

        Returns:
            results: Dictionary containing lists of training losses, test losses, and test accuracies
        """
        results = {"ssl_train_loss": [], "ssl_val_loss": []}
        
        for epoch in range(self.epochs_pt):
            ssl_train_loss = self.train_step(self.train_loader)
            ssl_val_loss = self.val_step(self.valcont_loader)

            results["ssl_train_loss"].append(ssl_train_loss)
            results["ssl_val_loss"].append(ssl_val_loss)

            if ssl_val_loss < self.best_val_loss:
                self.best_val_loss = ssl_val_loss
                self.save_checkpoint(epoch, ssl_train_loss, ssl_val_loss)
            print(f"Epoch {epoch+1}/{self.epochs_pt}, SSL Train Loss: {ssl_train_loss:.4f}, SSL Val Loss: {ssl_val_loss:.4f}")

        ft_trainer = self.finetune_step()
        ft_results = ft_trainer.train()
        
        # Update results with fine-tuning metrics
        results.update(ft_results)
        
        return results
    