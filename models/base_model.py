import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, base_encoder, input_shape=None, feature_size=None, output_shape=None, hidden_units=1024, proj_units=128):
        super().__init__()
        self.encoder = base_encoder

        self.hidden_units = hidden_units
        self.proj_units = proj_units

        # Get the output dimension from the last layer
        if feature_size is not None:
            self.feature_size = feature_size
        elif input_shape is not None:
            self.feature_size = self.encoder.calc_feat_size(input_shape)
        else:
            raise ValueError("Either feature_size or input_shape must be provided")
        
        
        self.classifier_head = nn.Linear(in_features=self.feature_size, out_features=output_shape)

        # Add a projection head (MLP with one hidden layer)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units,  self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units,  self.proj_units),
        )
    def forward(self):
        """
        Base model.
        This is for reference only and should be implemented by specific models.

        Args:
            epochs: Number of epochs to train for

        Returns:
            train_losses: List of average losses for each epoch
        """
        raise NotImplementedError("Subclasses must implement the model")