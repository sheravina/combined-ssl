# combined-ssl
welcome to my thesis project's github repo! 

## getting started
to run the code:
```shell
python main.py --dataset [dataset_name] --encoder [encoder_name] --model [model_type] --ssl [ssl method]

without []

dataset_name : debug
encoder : vgg
model_type : supervised, unsupervised, combined
ssl_method : simclr
```

## future implementations

model : supervised learning , self-supervised learning, combined learning (supervised + ssl)

SSL methods : SimCLR✅, MoCo📝, SimSiam📝, Jigsaw Puzzle📝, VAE📝, SimMIM📝

backbone : VGG✅, Resnet-10🚧, Resnet-50🚧 and ViT-B🚧

dataset :  debug (10% CIFAR-10)✅, CIFAR-10🚧, CIFAR-100🚧, ImageNet🚧 and Caltech-101🚧
