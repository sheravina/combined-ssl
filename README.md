# combined-ssl
welcome to my thesis project's github repo! 

## getting started
to run the code:
```shell
python main.py --dataset [dataset_name] --encoder [encoder_name] --model [model_type] --ssl [ssl method] --opt [optimizer name] --bs [batch size] --eppt [pretrain epochs] --epft [finetune epochs] --lr [learning rate] --wd [weight decay] --seed [seed]

without []

dataset_name : debug, cifar10
encoder : vgg, resnet18
model_type : supervised, unsupervised, combined
ssl_method : simclr, simsiam, vicreg, rotation
opt: sgd, lars, adam
bs: 128, 256, 512
lr: float
wd: float
seed: int
```

## future implementations

models : supervised learning , self-supervised learning, combined learning (supervised + ssl)

SSL methods : SimCLR✅, Jigsaw Puzzle✅, SimSiam✅, VICReg✅

encoders : Resnet-18✅, Resnet-50✅, Resnet-101✅, MobileNetv3📝, InceptionNet📝 and TinyViT📝

datasets :  debug (10% CIFAR-10)✅, CIFAR-10✅, CIFAR-100✅, Caltech-101✅ and TinyImageNet📝

📝 Planned --> ⏳ Ongoing --> 🚧 Done but unchecked (internal) --> ✅ Done and checked (internal) --> 💯 Approved (external)