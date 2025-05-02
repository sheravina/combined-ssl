# (debug, cifar10, cifar100, caltech101, imagenet)
DEBUG_DATASET = "debug"
CIFAR10_DATASET = "cifar10"
CIFAR100_DATASET = "cifar100"
CALTECH101_DATASET = "caltech101"
IMAGENET_DATASET = "imagenet"

# SSL Method simclr, moco, simsiam, jigsaw, vae, simmim
SSL_SIMCLR = "simclr"
SSL_JIGSAW = "jigsaw"
SSL_SIMSIAM = "simsiam"
SSL_VICREG = "vicreg"
SSL_ROTATION = "rotation"

# Encoder  (vgg, resnet10, resnet50, vit)
ENC_VGG = "vgg"
ENC_RESNET18 = "resnet18"
ENC_RESNET50 = "resnet50"
ENC_RESNET50_PT = "resnet50pt"
ENC_RESNET101 = "resnet101"
ENC_MNETV3 = "mnet"
ENC_INET = "inet"
ENC_VIT = "vit"
ENC_TINYVIT = "timmtinyvit"
ENC_VIT_TINY = "vit_tiny"
ENC_VIT_SMALL = "vit_small"
ENC_VIT_BASE = "vit_base"
ENC_CUSTOMRESNET = "custom"


# Model Name

MOD_SUPERVISED = "supervised"
MOD_UNSUPERVISED = "unsupervised"
MOD_COMBINED = "combined"
MOD_FINETUNED = "finetuned"

# Optimizers

OPT_ADAM = "adam"
OPT_LARS = "lars"
OPT_SGD = "sgd"