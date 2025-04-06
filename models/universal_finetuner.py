import torch.nn as nn

class UniversalFineTuner(nn.Module):
    def __init__(self, unsupervised_model, num_classes):
        super(UniversalFineTuner, self).__init__()
        self.encoder = unsupervised_model.encoder
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(unsupervised_model.feature_dim *unsupervised_model.feature_dim *num_classes, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x