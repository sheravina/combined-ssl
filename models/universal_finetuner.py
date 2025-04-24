import torch.nn as nn

class UniversalFineTuner(nn.Module):
    def __init__(self, unsupervised_model, output_shape = None):
        super(UniversalFineTuner, self).__init__()
        self.encoder = unsupervised_model.encoder
        if output_shape == None:
            self.classifier_head = nn.Linear(self.encoder.calc_feat_size(unsupervised_model.ft_input_shape), unsupervised_model.output_shape)
        else:
            self.classifier_head = nn.Linear(self.encoder.calc_feat_size(unsupervised_model.ft_input_shape), output_shape)
        
    def forward(self, x):
        h = self.encoder(x)
        c = self.classifier_head(h)
        return c