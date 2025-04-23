import os
import torch
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from .base_trainer import BaseTrainer


from trainers import BaseUnsupervisedTrainer

class SimCLRTrainer(BaseUnsupervisedTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, test_loader, val_loader, ft_loader, valcont_loader, optimizer, optimizer_name, lr_scheduler, lr, weight_decay, epochs, epochs_pt, epochs_ft, save_dir=None, temperature=0.5):
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
        self.temperature = temperature

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
        
        train_loss = train_loss / len(dataloader)

        try:
            self.lr_scheduler.step()
        except:
            self.lr_scheduler.step(train_loss)  
            
        return train_loss
    
    # def val_step(self, dataloader):

    #     val_loss = 0
    #     self.model.eval()
    #     with torch.no_grad():
    #         for batch, (images, _) in enumerate(dataloader):
    #             # Similar to train_step but without backprop
    #             img_1, img_2 = images[0].to(self.device), images[1].to(self.device)
    #             b, c, h, w = images[0].shape
    #             input_ = torch.cat([img_1.unsqueeze(1), img_2.unsqueeze(1)], dim=1)
    #             input_ = input_.view(-1, c, h, w)
    #             input_ = input_.cuda(non_blocking=True)
    #             output = self.model(input_).view(b, 2, -1)

    #             loss = simclr_loss(output, self.temperature)
    #             val_loss += loss.item()
        
    #     val_loss = val_loss / len(dataloader)

    #     return val_loss
    
    # def finetune_step(self):
    #     # Load the best model if it exists
    #     best_model_path = os.path.join(self.save_dir, 'best_ssl_model.pth')
    #     if os.path.exists(best_model_path):
    #         checkpoint = torch.load(best_model_path)
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #         print(f"Loaded best SSL model from epoch {checkpoint['epoch']+1} for finetuning")
    #     else:
    #         print("No best model found, using current model for finetuning")
        
    #     ft_model = UniversalFineTuner(self.model).to(self.device)
    #     # Only train the classifier layer
    #     ft_optimizer = optim.Adam(ft_model.classifier_head.parameters(), lr=1e-4)
    #     ft_trainer = SupervisedTrainer(model=ft_model
    #                                         ,train_loader=self.ft_loader
    #                                         ,test_loader=self.test_loader
    #                                         ,val_loader=self.val_loader
    #                                         ,ft_loader = None
    #                                         ,optimizer=ft_optimizer
    #                                         ,lr_scheduler=self.lr_scheduler
    #                                         ,epochs=self.epochs_ft
    #                                         ,save_dir=self.save_dir)
        
    #     return ft_trainer
    
    # def save_checkpoint(self, epoch, ssl_loss):
    #     """
    #     Save model checkpoint - overwrites the previous best model
        
    #     Args:
    #         epoch: Current epoch
    #         ssl_loss: SSL training loss
    #         ft_val_loss: Finetuning validation loss (optional)
    #         ft_train_loss: Finetuning training loss (optional)
    #         ft_val_acc: Finetuning validation accuracy (optional)
    #         ft_train_acc: Finetuning training accuracy (optional)
    #     """            
    #     # Use a fixed filename for the best model
    #     best_model_path = os.path.join(self.save_dir, 'best_ssl_model.pth')
        
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.lr_scheduler.state_dict(),
    #         'ssl_loss': ssl_loss
    #     }
            
    #     torch.save(checkpoint, best_model_path)
        
    #     print(f'New best SimCLR model saved to {best_model_path} (Epoch {epoch+1}, SSL Loss: {ssl_loss:.4f})')

    
    # def train(self):
    #     """
    #     Trains and tests the model for the specified number of epochs.

    #     Returns:
    #         results: Dictionary containing lists of training losses, test losses, and test accuracies
    #     """
    #     results = {"ssl_train_loss": [], "ssl_val_loss": []}
        
    #     for epoch in range(self.epochs_pt):
    #         ssl_train_loss = self.train_step(self.train_loader)
    #         ssl_val_loss = self.val_step(self.valcont_loader)

    #         results["ssl_train_loss"].append(ssl_train_loss)
    #         results["ssl_val_loss"].append(ssl_val_loss)

    #         if ssl_val_loss < self.best_val_loss:
    #             self.best_val_loss = ssl_val_loss
    #             self.save_checkpoint(epoch, ssl_train_loss, ssl_val_loss)
    #         print(f"Epoch {epoch+1}/{self.epochs_pt}, SSL Train Loss: {ssl_train_loss:.4f}, SSL Val Loss: {ssl_val_loss:.4f}")

    #     ft_trainer = self.finetune_step()
    #     ft_results = ft_trainer.train()
        
    #     # Update results with fine-tuning metrics
    #     results.update(ft_results)
        
    #     return results