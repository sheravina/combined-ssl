import torch
from tqdm import tqdm
from losses.supervised_loss import SupervisedLoss
from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised learning."""

    def __init__(self, model, train_loader, optimizer, epochs):
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
        super().__init__(model, train_loader, optimizer, epochs)
        self.criterion = SupervisedLoss()

    def train(self):
        """
        Train the model using supervised learning.

        Returns:
            train_losses: List of average losses for each epoch
        """
        self.model.train()
        train_losses = []
        print("model is training ....")

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_steps = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in pbar:
                images, labels = batch

                # Get the original images (assuming third item is the original)
                img = (
                    images[2].to(self.device)
                    if len(images) > 2
                    else images[0].to(self.device)
                )
                labels = labels.to(self.device)

                # Forward pass for classification
                pred = self.model(img)  # Assuming model returns predictions

                # Compute supervised loss
                loss = self.criterion(pred, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                epoch_steps += 1
                pbar.set_postfix({"loss": epoch_loss / epoch_steps})

            avg_loss = epoch_loss / epoch_steps
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        return train_losses