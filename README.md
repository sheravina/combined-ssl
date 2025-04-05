to run the code:

python main.py --d --r --e --

current structure of the code

├── aux-losses
├── data
├── encoders
│   ├── __init__.py
│   ├── base_encoder.py : class BaseEncoder
│   ├── resnet.py: class ResNetEncoder that inherits BaseEncoder
│   ├── vgg.py: class VGGEncoder that inherits BaseEncoder
│   └── vit.py: class ViTEncoder that inherits BaseEncoder
├── losses
│   ├── __init__.py
│   ├── simclr_loss.py
│   └── supervised_loss.py
├── models
│   ├── __init__.py
│   ├── combined_simclr.py : class CombinedSimCLR
│   ├── supervised.py : class SupervisedModel
│   └── unsupervised_simclr.py : class SimCLR
└── README.md

geplante Experimente

folder: .py files
model: supervised learning , self-supervised learning, combined learning (supervised + )
--SSL methods: SimCLR, MoCo, SimSiam, Jigsaw Puzzle, VAE, SimMIM
backbone: VGG (for testing purposes), Resnet-10, Resnet-50 and ViT-B
dataset:  10% CIFAR-10 (for testing purposes), CIFAR-10, CIFAR-100, ImageNet and Caltech-101
losses: every contrastive losses plus combined losses



batch size:

CIFAR-10/100: 256
Caltech-101: 128
ImageNet: 512 (or largest possible on your cluster)
Ablation study: Include a section testing different batch sizes (128, 256, 512, 1024) 


Training Epochs:

300 epochs for pre-training, 100 epochs for fine-tuning, 
300 epochs for combined and supervised

Learning rates: 



what stays the same


