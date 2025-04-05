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
        super().__init__(model, train_loader, optimizer, epochs)
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = SupervisedLoss()

    def train(self):
        """
        Train the model using combined supervised and contrastive learning.

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
                images, labels = batch
                # Get the contrastive pairs and original image
                img_1, img_2, img_3 = (
                    images[0].to(self.device),
                    images[1].to(self.device),
                    images[2].to(self.device),
                )
                b, c, h, w = images[0].shape

                # Prepare the input for contrastive loss
                input_ = torch.cat([img_1.unsqueeze(1), img_2.unsqueeze(1)], dim=1)
                input_ = input_.view(-1, c, h, w)
                input_ = input_.cuda(non_blocking=True)

                # Forward pass for contrastive learning
                _, output = self.model(input_)
                output = output.view(b, 2, -1)

                # Compute contrastive loss
                contrastive_loss = simclr_loss(output, self.temperature)

                # Forward pass for supervised learning
                labels = labels.to(self.device)
                pred, _ = self.model(img_3)

                # Compute supervised loss on the original image
                supervised_loss = self.criterion(pred, labels)

                # Combined loss
                loss = contrastive_loss + self.alpha * supervised_loss

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                epoch_steps += 1
                pbar.set_postfix(
                    {
                        "total_loss": loss.item(),
                        "contrastive": contrastive_loss.item(),
                        "supervised": supervised_loss.item(),
                    }
                )

            avg_loss = epoch_loss / epoch_steps
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        return train_losses
