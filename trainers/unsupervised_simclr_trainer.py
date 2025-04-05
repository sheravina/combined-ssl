import torch
from tqdm import tqdm
from losses.simclr_loss import simclr_loss
from .base_trainer import BaseTrainer


class SimCLRTrainer(BaseTrainer):
    """Trainer for self-supervised contrastive learning (SimCLR)."""

    def __init__(self, model, train_loader, optimizer, epochs, temperature=0.5):
        """
        Initialize the SimCLR trainer.

        Args:
            model: The contrastive learning model
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            device: Device to run training on ('cuda' or 'cpu')
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__(model, train_loader, optimizer, epochs)
        self.temperature = temperature

    def train(self):
        """
        Train the model using contrastive learning.

        Args:
            epochs: Number of epochs to train for

        Returns:
            train_losses: List of average losses for each epoch
        """
        self.model.train()
        train_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_steps = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in pbar:
                images, _ = batch
                # Get the contrastive pairs
                img_1, img_2 = images[0].to(self.device), images[1].to(self.device)
                b, c, h, w = images[0].shape

                # Prepare the input for the model
                input_ = torch.cat([img_1.unsqueeze(1), img_2.unsqueeze(1)], dim=1)
                input_ = input_.view(-1, c, h, w)
                input_ = input_.cuda(non_blocking=True)

                # Forward pass
                output = self.model(input_).view(b, 2, -1)

                # Compute contrastive loss
                loss = simclr_loss(output, self.temperature)

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
