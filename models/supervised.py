from .base_model import BaseModel
class SupervisedModel(BaseModel):
    def __init__(self, base_encoder, input_shape=None, feature_size=None, output_shape=None, hidden_units=1024, proj_units=128):
        # Fix: Pass the actual parameters to the parent class
        super().__init__(base_encoder=base_encoder, input_shape=input_shape, feature_size=feature_size, output_shape=output_shape, hidden_units=hidden_units, proj_units=proj_units)

    def forward(self, x):
        # Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        # Process through encoder
        h = self.encoder(x)
        # Process through classifier head
        c = self.classifier_head(h)
        return c