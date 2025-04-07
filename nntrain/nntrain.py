# TODO: add seeds, batch size, epochs, learning rate, weight decay to somewhere
import torch
import torch.optim as optim
from data import DataManager
from utils.constants import *
from pprint import pprint
from encoders import *
from models import *
from trainers import *
from torchinfo import summary

class NNTrain:
    def __init__(self, dataset_name: str, ssl_method: str, encoder_name: str, model_name:str) -> None:
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.encoder_name = encoder_name
        self.model_name = model_name

        # still hard coded!
        self.batch_size = 32 
        self.epochs = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.seed = 42

        self.load_data()
        self.data_char()
        self.init_encoder()
        self.init_model_trainer()

    def load_data(self):
        # check dataset name
        dm = DataManager(dataset_name=self.dataset_name, ssl_method=self.ssl_method, batch_size=self.batch_size, seed=self.seed)
        train_loader, cont_loader, test_loader = dm.create_loader()
        self.train_loader = train_loader
        self.cont_loader = cont_loader
        self.test_loader = test_loader

    def data_char(self):
        # Grab data characteristics
        images, labels = next(iter(self.train_loader))
        self.input_shape = images[0].shape
        self.output_shape = len(torch.unique(labels))
        self.color_channels = self.input_shape[1]
    
    def init_encoder(self):
        if self.encoder_name == ENC_VGG:
            self.encoder = VGGEncoder()

        elif self.encoder_name == ENC_RESNET18: 
            self.encoder = ResNetEncoder(model_type=ENC_RESNET18)
        
        elif self.encoder_name == ENC_RESNET50:
            self.encoder = ResNetEncoder(model_type=ENC_RESNET50)

        elif self.encoder_name == ENC_VIT:
            self.encoder = ViTEncoder()
    
    def init_model_trainer(self):
        if self.model_name == MOD_SUPERVISED:
            self.model = SupervisedModel(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = SupervisedTrainer(model=self.model, train_loader=self.train_loader, test_loader=self.test_loader, ft_loader=None, optimizer=self.optimizer, epochs=self.epochs)
            
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_SIMCLR:
            self.model = SimCLR(base_encoder=self.encoder, input_shape = self.input_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = SimCLRTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=self.train_loader, optimizer=self.optimizer, epochs=self.epochs)
        
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMCLR:
            self.model = CombinedSimCLR(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = CombinedSimCLRTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=None, optimizer=self.optimizer, epochs=self.epochs)
            
        self.trainer.train()
        
