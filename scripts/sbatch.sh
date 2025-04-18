#!/bin/sh
#SBATCH --mail-user=sheravina@master.ismll.de
#SBATCH --job-name=debug-run
#SBATCH --output=./log/py_torch_test%j.log
#SBATCH --error=./log/py_torch_test%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1



./../miniconda3/bin/activate SRP_ENV
srun python main.py --dataset cifar10 --encoder resnet18 --model supervised --ssl simclr --opt sgd --bs 128 --eppt 150 --epft 50 --lr 0.01 --wd 0.0005 --seed 42
