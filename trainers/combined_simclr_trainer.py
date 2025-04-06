import torch
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from losses.supervised_loss import SupervisedLoss
from .base_trainer import BaseTrainer


class CombinedSimCLRTrainer(BaseTrainer):
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
        alpha=0.5
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
        self.criterion = SupervisedLoss()

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
            img_1, img_2, img_3, labels = images[0].to(self.device), images[1].to(self.device), images[2].to(self.device), labels.to(self.device)
            b, c, h, w = images[0].shape
            input_ = torch.cat([img_1.unsqueeze(1), img_2.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            _, output = self.model(input_)
            output = output.view(b, 2, -1)
            contrastive_loss = simclr_loss(output, self.temperature)
            
            # calculate supervised loss for the original image
            pred, _ = self.model(img_3)
            supervised_loss = self.criterion(pred, labels)

            # Combined loss
            loss = contrastive_loss + self.alpha * supervised_loss
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
                test_pred, _ = self.model(images)
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