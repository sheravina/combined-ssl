import torch
from tqdm import tqdm


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(self, model, train_loader, test_loader, optimizer, epochs):
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
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        """
        Base training method.
        This is for reference only and should be implemented by specific trainers.

        Args:
            epochs: Number of epochs to train for

        Returns:
            train_losses: List of average losses for each epoch
        """
        raise NotImplementedError("Subclasses must implement the train method")
