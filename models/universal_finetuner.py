import torch.nn as nn

class UniversalFineTuner(nn.Module):
    def __init__(self, unsupervised_model):
        super(UniversalFineTuner, self).__init__()
        self.encoder = unsupervised_model.encoder
        self.classifier_head = nn.Linear(unsupervised_model.feature_size, unsupervised_model.output_shape)
        
    def forward(self, x):
        h = self.encoder(x)
        c = self.classifier_head(h)
        return c