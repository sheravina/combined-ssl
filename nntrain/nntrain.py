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
from .lars import LARS

class NNTrain:
    def __init__(self
                 , dataset_name: str, ssl_method: str, encoder_name: str, model_name:str, save_toggle:bool
                 , optimizer_name, batch_size, epochs_pt, epochs_ft, learning_rate, weight_decay, seed, jname ) -> None:
        self.dataset_name = dataset_name
        self.ssl_method = ssl_method
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.save_toggle = save_toggle

        # still hard coded!
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.epochs_pt = epochs_pt
        self.epochs_ft = epochs_ft
        self.epochs = self.epochs_pt + self.epochs_ft
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.jname = jname

        # Create run directory with timestamp for this training session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results/{self.timestamp}_{self.jname}_{self.dataset_name}_{self.encoder_name}_{self.model_name}_{self.ssl_method}_{self.batch_size}_{self.seed}"
        self.checkpoint_dir = f"{self.results_dir}/checkpoints"
        
        # Create the directories if they don't exist
        if self.save_toggle:
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_time = time.time()

        self.load_data()
        self.data_char()
        self.init_encoder()
        self.init_model_trainer()

        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time
        self.training_time_formatted = str(timedelta(seconds=self.training_time))

        if self.save_toggle:
            self.save_results_to_excel()

    def load_data(self):
        # check dataset name
        dm = DataManager(dataset_name=self.dataset_name, ssl_method=self.ssl_method, batch_size=self.batch_size, seed=self.seed, encoder=self.encoder_name)
        train_loader, cont_loader, test_loader, val_loader, valcont_loader = dm.create_loader()
        self.train_loader = train_loader
        self.cont_loader = cont_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.valcont_loader = valcont_loader

    def data_char(self):
        # Grab data characteristics
        try:
            images_cont, labels_cont = next(iter(self.cont_loader))
        except:
            images, images_cont, labels_cont, labels = next(iter(self.cont_loader))
        images_train, labels_train = next(iter(self.train_loader))
        self.input_shape = images_train[0].shape
        self.input_shape_jigsaw = images_cont[0][0][0].shape
        # self.input_shape_ftrot = images_cont[0].shape
        try:
            self.output_shape_ftrot = len(torch.unique(labels))
        except:
            pass
        self.color_channels = self.input_shape[1]
        self.output_shape = len(torch.unique(labels_cont))
        
    
    def init_encoder(self):
        if self.encoder_name == ENC_VGG:
            self.encoder = VGGEncoder()

        elif self.encoder_name in [ENC_RESNET18, ENC_RESNET50, ENC_RESNET101, ENC_RESNET50_PT]:
            self.encoder = ResNetEncoder(model_type=self.encoder_name)

        elif self.encoder_name == ENC_VIT_TINY:
            self.encoder = ViTEncoder(ENC_VIT_TINY)

        elif self.encoder_name == ENC_MNETV3:
            self.encoder = MobileNetV3()
        
        elif self.encoder_name == ENC_INET:
            self.encoder = InceptionNet()
        
        elif self.encoder_name == ENC_TINYVIT:
            self.encoder = TinyViTEncoder(ENC_TINYVIT)
        
        elif self.encoder_name == ENC_CUSTOMRESNET:
            self.encoder = CustomResNet()
        
    
    def init_model_trainer(self):

        if self.model_name == MOD_SUPERVISED:
            self.model = SupervisedModel(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        if self.model_name == MOD_FINETUNED:
            self.model = SupervisedModel(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_SIMCLR:
            self.model = SimCLR(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMCLR:
            self.model = CombinedSimCLR(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape)

        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_SIMSIAM:
            self.model = SimSiam(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMSIAM:
            self.model = CombinedSimSiam(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape)       

        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_VICREG:
            self.model = VICReg(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_VICREG:
            self.model = CombinedVICReg(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape) 
        
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_ROTATION:
            self.model = Rotation(base_encoder=self.encoder, input_shape = self.input_shape, output_shape = self.output_shape)

        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_ROTATION:
            self.model = CombinedRotation(base_encoder=self.encoder, input_shape = self.input_shape, output_shape=self.output_shape
                                          ,output_shape_ssl=self.output_shape, output_shape_sup=self.output_shape_ftrot)

        # print(f"\nModel Summary for {self.model_name} with {self.encoder_name}:")
        # summary(self.model)
        
        # # Safely find and print kernel information by searching through all modules
        # print("\nConvolutional layer kernels:")
        # found_conv = False
        # for name, module in self.model.named_modules():
        #     if isinstance(module, torch.nn.Conv2d):
        #         found_conv = True
        #         print(f"Layer: {name}")
        #         print(f"  Kernel shape: {module.weight.shape}")
        #         print(f"  Sample kernel values (first filter, first channel):")
        #         print(f"  {module.weight[0, 0].detach().cpu().numpy()}")
        #         # Only print details for the first few conv layers
        #         if name.count('.') > 2:  # Skip printing too many layers
        #             print("  ...")
        #             break
        
        # if not found_conv:
        #     print("No convolutional layers found in the model.")

        # pretrained_path = os.path.join('pt_models', 'simclr_resnet50.pth')
        # if os.path.exists(pretrained_path):
        #     print(f"Loading pretrained model from {pretrained_path}")
        #     checkpoint = torch.load(pretrained_path, weights_only=True)
        #     state_dict = checkpoint['state_dict']
        #     # for k in list(state_dict.keys()):
        #     #     # retain only encoder up to before the embedding layer
        #     #     if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
        #     #         # remove prefix
        #     #         state_dict[k[len("module.encoder."):]] = state_dict[k]
        #     #     # delete renamed or unused k
        #     #     del state_dict[k]
        #     self.model.load_state_dict(state_dict)
        #     print("Pretrained model loaded successfully")
        # else:
        #     print(f"Warning: Pretrained model not found at {pretrained_path}. Training from scratch.")


        if self.optimizer_name == OPT_ADAM:
            self.optimizer_selected = optim.AdamW(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == OPT_LARS:
            self.optimizer_selected = LARS(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

        elif self.optimizer_name == OPT_SGD:
            self.optimizer_selected = optim.SGD(self.model.parameters(),lr=self.learning_rate,momentum=0.9, weight_decay=self.weight_decay, nesterov=True)


        self.lr_scheduler_selected = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_selected,T_max=self.epochs_pt if self.model_name == MOD_UNSUPERVISED else self.epochs, eta_min = 0) #0.001
        # self.lr_scheduler_selected = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer_selected,mode='max',verbose=True,factor=0.1,patience=3,threshold=0.001)


        if self.model_name == MOD_SUPERVISED:
            self.trainer = SupervisedTrainer(model=self.model
                                            , train_loader=self.train_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                            , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                            , epochs=self.epochs
                                            , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        if self.model_name == MOD_FINETUNED:
            self.trainer = FTSupervisedTrainer(model=self.model
                                            , train_loader=self.train_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                            , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                            , epochs=self.epochs
                                            , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_SIMCLR:
            self.trainer = SimCLRTrainer(model=self.model
                                         , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=self.train_loader, valcont_loader = self.valcont_loader
                                         , optimizer=self.optimizer_selected, optimizer_name=self.optimizer_name, lr_scheduler = self.lr_scheduler_selected,lr=self.learning_rate, weight_decay=self.weight_decay
                                         , epochs=self.epochs, epochs_pt = self.epochs_pt, epochs_ft = self.epochs_ft
                                         , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMCLR:
            self.trainer = CombinedSimCLRTrainer(model=self.model
                                                 , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                                 , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                                 , epochs=self.epochs
                                                 , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_SIMSIAM:
            self.trainer = SimSiamTrainer(model=self.model
                                          , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=self.train_loader, valcont_loader = self.valcont_loader
                                          , optimizer=self.optimizer_selected, optimizer_name=self.optimizer_name, lr_scheduler = self.lr_scheduler_selected,lr=self.learning_rate, weight_decay=self.weight_decay
                                          , epochs=self.epochs, epochs_pt = self.epochs_pt, epochs_ft = self.epochs_ft
                                          , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_SIMSIAM:
            self.trainer = CombinedSimSiamTrainer(model=self.model
                                                  , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                                  , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                                  , epochs=self.epochs
                                                  , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_VICREG:
            self.trainer = VICRegTrainer(model=self.model
                                         , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=self.train_loader, valcont_loader = self.valcont_loader
                                         , optimizer=self.optimizer_selected, optimizer_name=self.optimizer_name, lr_scheduler = self.lr_scheduler_selected,lr=self.learning_rate, weight_decay=self.weight_decay
                                         , epochs=self.epochs, epochs_pt = self.epochs_pt, epochs_ft = self.epochs_ft
                                         , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_VICREG:
            self.trainer = CombinedVICRegTrainer(model=self.model
                                                 , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                                 , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                                 , epochs=self.epochs
                                                 , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_UNSUPERVISED and self.ssl_method == SSL_ROTATION:
            self.trainer = RotTrainer(model=self.model
                                         , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=self.train_loader, valcont_loader = self.valcont_loader
                                         , optimizer=self.optimizer_selected, optimizer_name=self.optimizer_name, lr_scheduler = self.lr_scheduler_selected,lr=self.learning_rate, weight_decay=self.weight_decay
                                         , epochs=self.epochs, epochs_pt = self.epochs_pt, epochs_ft = self.epochs_ft, ft_output = self.output_shape_ftrot
                                         , save_dir=self.checkpoint_dir if self.save_toggle else None)
            
        elif self.model_name == MOD_COMBINED and self.ssl_method == SSL_ROTATION:
            self.trainer = CombinedRotTrainer(model=self.model
                                                 , train_loader=self.cont_loader, test_loader=self.test_loader, val_loader=self.val_loader, ft_loader=None
                                                 , optimizer=self.optimizer_selected, lr_scheduler = self.lr_scheduler_selected
                                                 , epochs=self.epochs
                                                 , save_dir=self.checkpoint_dir if self.save_toggle else None)
    
        self.results = self.trainer.train()

    def save_results_to_excel(self):
        # Create a dictionary with parameters for the first sheet
        params_data = {
            'dataset_name': self.dataset_name,
            'ssl_method': self.ssl_method,
            'encoder_name': self.encoder_name,
            'model_name': self.model_name,
            'optimizer' : self.optimizer_name,
            'batch_size': self.batch_size,
            'epochs_pt' : self.epochs_pt,
            'epochs_ft' : self.epochs_ft,
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

        # Create a DataFrame for results
        results_df = pd.DataFrame(results_rows)
        
        # Define Excel file path in the same directory as checkpoints
        excel_path = f"{self.results_dir}/training_results_{self.dataset_name}_{self.encoder_name}_{self.model_name}_{self.ssl_method}_{self.batch_size}_{self.seed}_{self.timestamp}.xlsx"
        # self.results_dir = f"results/{self.dataset_name}_{self.encoder_name}_{self.model_name}_{self.ssl_method}_{self.batch_size}_{self.seed}_{self.timestamp}"
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write each dataframe to a different worksheet
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            results_df.to_excel(writer, sheet_name='Results', index=False)
        
        # print(f"Results saved to {excel_path}")
        # print(f"Model checkpoints saved to {self.checkpoint_dir}")