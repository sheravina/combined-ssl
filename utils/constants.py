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

# Encoder  (vgg, resnet10, resnet50, vit)
ENC_VGG = "vgg"
ENC_RESNET18 = "resnet18"
ENC_RESNET50 = "resnet50"
ENC_RESNET101 = "resnet101"
ENC_VIT = "vit"

# Model Name

MOD_SUPERVISED = "supervised"
MOD_UNSUPERVISED = "unsupervised"
MOD_COMBINED = "combined"

# Optimizers

OPT_ADAM = "adam"
OPT_LARS = "lars"
OPT_SGD = "sgd"