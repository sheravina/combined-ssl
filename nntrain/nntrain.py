# TODO: hard coded variables be moved soon
import os
import torch
import torch.optim as optim
import pandas as pd
from data import DataManager
from utils.constants import *
from pprint import pprint
from encoders import *
from models import *
from trainers import *
from torchinfo import summary
import time
from datetime import datetime,timedelta

class NNTrain:
    def __init__(self, dataset_name: str, ssl_method: str, encoder_name: str, model_name:str, save_toggle:bool) -> None:
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.save_toggle = save_toggle

        # still hard coded!
        self.batch_size = 32 
        self.epochs = 2 
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.seed = 42

        self.start_time = time.time()

        self.load_data()
        self.data_char()
        self.init_encoder()
        self.init_model_trainer()

        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time
        self.training_time_formatted = str(timedelta(seconds=self.training_time))

        self.save_results_to_excel()

    def load_data(self):
        # check dataset name
        dm = DataManager(dataset_name=self.dataset_name, ssl_method=self.ssl_method, batch_size=self.batch_size, seed=self.seed)
        train_loader, cont_loader, test_loader = dm.create_loader()
        self.train_loader = train_loader
        self.cont_loader = cont_loader
        self.test_loader = test_loader

    def data_char(self):
        # Grab data characteristics
        images_cont, labels_cont = next(iter(self.cont_loader))
        images_train, labels_train = next(iter(self.train_loader))
        self.input_shape = images_train[0].shape
        self.input_shape_jigsaw = images_cont[0][0][0].shape
        self.color_channels = self.input_shape[1]
        self.output_shape = len(torch.unique(labels_cont))
        
    
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
            self.model = SimCLR(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = SimCLRTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=self.train_loader, optimizer=self.optimizer, epochs=self.epochs)
        
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMCLR:
            self.model = CombinedSimCLR(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = CombinedSimCLRTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=None, optimizer=self.optimizer, epochs=self.epochs)

        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_JIGSAW:
            self.model = Jigsaw(base_encoder=self.encoder, input_shape = self.input_shape_jigsaw, ft_input_shape=self.input_shape, output_shape = self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = JigsawTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=self.train_loader, optimizer=self.optimizer, epochs=self.epochs)
        
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_JIGSAW:
            self.model = CombinedJigsaw(base_encoder=self.encoder, input_shape = self.input_shape_jigsaw, ft_input_shape=self.input_shape, output_shape = self.output_shape)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)
            self.trainer = CombinedJigsawTrainer(model=self.model, train_loader=self.cont_loader, test_loader=self.test_loader, ft_loader=self.train_loader, optimizer=self.optimizer, epochs=self.epochs)
            
        self.results = self.trainer.train()

    def save_results_to_excel(self):

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Create a dictionary with parameters for the first sheet
        params_data = {
            'dataset_name': self.dataset_name,
            'ssl_method': self.ssl_method,
            'encoder_name': self.encoder_name,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'seed': self.seed,
            'training_time_seconds': self.training_time,
            'training_time_formatted': self.training_time_formatted,
        }
        
        # Create a DataFrame for parameters
        params_df = pd.DataFrame([params_data])
        
        # Create a dictionary with results for the second sheet
        results_rows = []
        if hasattr(self, 'results') and self.results:
            # First, get all the keys and determine the number of epochs
            keys = list(self.results.keys())
            if keys and isinstance(self.results[keys[0]], list):
                num_epochs = len(self.results[keys[0]])
                
                # Create a row for each epoch
                for epoch_idx in range(num_epochs):
                    row_data = {'epoch': epoch_idx + 1}  # 1-based epoch numbering
                    
                    # Add the value for each metric at this epoch
                    for key in keys:
                        if isinstance(self.results[key], list) and epoch_idx < len(self.results[key]):
                            row_data[key] = self.results[key][epoch_idx]
                    
                    results_rows.append(row_data)

        if self.save_toggle == True:
            # Create a DataFrame for results
            results_df = pd.DataFrame(results_rows)
            
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results_dir}/{self.dataset_name}_{self.encoder_name}_{self.model_name}_{self.ssl_method}_{timestamp}.xlsx"
            
            # Create a Pandas Excel writer using XlsxWriter as the engine
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write each dataframe to a different worksheet
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                results_df.to_excel(writer, sheet_name='Results', index=False)
            
            print(f"Results saved to {filename}")
        else:
            print(f"save_toggle is OFF")