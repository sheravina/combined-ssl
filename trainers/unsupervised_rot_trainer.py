import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from .base_trainer import BaseTrainer
from torchvision import transforms
from models import UniversalFineTuner, SupervisedModel
from trainers import SupervisedTrainer
from utils.constants import *
from nntrain import LARS

from trainers import BaseUnsupervisedTrainer

to_tensor = transforms.ToTensor()

class RotTrainer(BaseUnsupervisedTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, ft_output, save_dir=None, temperature=0.5):
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
        
        self.ft_output = ft_output
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_step(self, dataloader):
        """
        Trains the model for a single epoch

        Returns:
            train_loss
        """
        self.model.train()
        train_loss = 0
        train_acc = 0

        for batch, (img, rotated_img, rotation_labels, labels) in enumerate(dataloader):

            images = rotated_img.to(self.device)
            rot_labels = rotation_labels.to(self.device)

            pred = self.model(images)
            loss = self.criterion(pred, rot_labels)
            train_loss += loss.item() 
            
            pred_labels = pred.argmax(dim=1)
            train_acc += ((pred_labels == rot_labels).sum().item()/len(pred_labels))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)  
        
        return train_loss
    
    def finetune_step(self):
        # Load the best model if it exists
        best_model_path = os.path.join(self.save_dir, 'best_ssl_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best SSL model from epoch {checkpoint['epoch']} for finetuning")
        else:
            print("No best model found, using current model for finetuning")
        
        ft_model = UniversalFineTuner(self.model, output_shape=self.ft_output).to(self.device)
        # Only train the classifier layer

        if self.optimizer_name == OPT_ADAM:
            ft_optimizer = optim.Adam(ft_model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == OPT_LARS:
            ft_optimizer = LARS(ft_model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

        elif self.optimizer_name == OPT_SGD:
            ft_optimizer = optim.SGD(ft_model.parameters(),lr=self.learning_rate,momentum=0.9, weight_decay=self.weight_decay, nesterov=False)

        # ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(), lr=1e-4)

        ft_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=ft_optimizer,T_max=self.epochs_ft, eta_min = 0)

        ft_trainer = SupervisedTrainer(model=ft_model
                                            ,train_loader=self.ft_loader
                                            ,test_loader=self.test_loader
                                            ,val_loader=self.val_loader
                                            ,ft_loader = None
                                            ,optimizer=ft_optimizer
                                            ,lr_scheduler=ft_lr_scheduler
                                            ,epochs=self.epochs_ft
                                            ,save_dir=self.save_dir)
        
        return ft_trainer
    
    def save_checkpoint(self, epoch, ssl_train_loss):
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
        epoch_model_path = os.path.join(self.save_dir, f'ssl_model_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'ssl_train_loss' : ssl_train_loss
        }
            
        torch.save(checkpoint, epoch_model_path)
        
        print(f'New best SimCLR model saved to {epoch_model_path} (Epoch {epoch+1}, SSL Train Loss: {ssl_train_loss:.4f})')