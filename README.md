# combined-ssl
welcome to my thesis project's github repo! 

## getting started
to run the code:
```shell
python main.py --dataset [dataset_name] --encoder [encoder_name] --model [model_type] --ssl [ssl method] --opt [optimizer name] --bs [batch size] --eppt [pretrain epochs] --epft [finetune epochs] --lr [learning rate] --wd [weight decay] --seed [seed]

without []

dataset_name : debug
encoder : vgg
model_type : supervised, unsupervised, combined
ssl_method : simclr
```

## future implementations

models : supervised learning , self-supervised learning, combined learning (supervised + ssl)

SSL methods : SimCLR✅, Jigsaw Puzzle🚧, SimSiam🚧, VICReg🚧

encoders : VGG✅, Resnet-18✅, Resnet-50✅, Resner-101✅, and ViT-B🚧

datasets :  debug (10% CIFAR-10)✅, CIFAR-10✅, CIFAR-100🚧, ImageNet🚧 and Caltech-101🚧

📝 Planned --> ⏳ Ongoing --> 🚧 Done but unchecked (internal) --> ✅ Done and checked (internal) --> 💯 Approved (external)