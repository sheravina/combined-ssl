import torch
from torch.autograd import Variable
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from losses.supervised_loss import SupervisedLoss
from trainers import BaseCombinedTrainer


class CombinedJigsawTrainer(BaseCombinedTrainer):
    """Trainer for combined supervised and contrastive learning."""

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        ft_loader,
        optimizer,
        epochs,
        temperature=0.5,
        alpha=0.5,
        save_dir=None
    ):
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
        super().__init__(model, train_loader, test_loader, ft_loader, optimizer, epochs)
        self.temperature = temperature
        self.alpha = alpha
        self.criterion_ssl = torch.nn.CrossEntropyLoss()
        self.criterion = SupervisedLoss()

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
            #images, labels = images.to(self.device), labels.to(self.device)
            # calculate contrastive loss between two augmented images
            permuted_tiles, permutation_idx, original_tiles, original_img = images[0].to(self.device), images[1].to(self.device),images[2].to(self.device), images[3].to(self.device)
            labels = labels.to(self.device)

            output, _ = self.model(permuted_tiles)
            logits = Variable(output.argmax(dim=1).float(),requires_grad = True)
            permutation_idx = Variable(permutation_idx.float(),requires_grad = True)
            contrastive_loss = self.criterion_ssl(logits, permutation_idx)
            ssl_loss += contrastive_loss.item()
            
            # calculate supervised loss for the original image
            _, pred = self.model(original_tiles)
            supervised_loss = self.criterion(pred, labels)
            sup_loss += supervised_loss.item()

            # Combined loss
            loss = contrastive_loss + self.alpha * supervised_loss
            train_loss += loss.item()

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
                permuted_tiles, permutation_idx, original_tiles, original_img = images[0].to(self.device), images[1].to(self.device),images[2].to(self.device), images[3].to(self.device)
                labels = labels.to(self.device)
                _, test_pred = self.model(original_tiles)
                loss = self.criterion(test_pred, labels)
                test_loss += loss.item()

                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
            
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc