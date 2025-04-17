# adapted from simsiam model implementation from https://github.com/facebookresearch/simsiam
import os
import torch
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from losses.supervised_loss import SupervisedLoss
from trainers import BaseCombinedTrainer


class CombinedSimSiamTrainer(BaseCombinedTrainer):
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
        self.criterion_ssl = torch.nn.CosineSimilarity(dim=1).to(self.device)
        self.criterion_sup = SupervisedLoss()
        self.save_dir = save_dir
        self.best_val_loss = float('inf')

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
            p1, p2, z1, z2, pred = self.model(x1=img_1, x2=img_2, x3=img_3)
            contrastive_loss = -(self.criterion_ssl(p1, z2).mean() + self.criterion_ssl(p2, z1).mean()) * 0.5
            ssl_loss += contrastive_loss.item()
            supervised_loss = self.criterion_sup(pred, labels)
            sup_loss += supervised_loss.item()

            # Combined loss
            loss = ssl_loss + self.alpha * supervised_loss
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
                _, _, _, _, test_pred = self.model(images, images, images)
                loss = self.criterion_sup(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc