from configparser import ConfigParser
import argparse
from nntrain import NNTrain
from utils.constants import DEBUG_DATASET
from pprint import pprint

if __name__ == "__main__":
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="select a dataset (debug, cifar10, cifar100, caltech101, imagenet)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="select an encoder (vgg, resnet10, resnet50, vit)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="select a model (supervised, unsupervised, combined)",
    )
    parser.add_argument(
        "--ssl",
        default="simclr",
        type=str,
        help="select a ssl method (simclr, simsiam, jigsaw, vicreg)",
    )  # ignored if model is supervised

    parser.add_argument(
        "--opt",
        type=str,
        help="select an optimizer (sgd, lars, adam)"
    )

    parser.add_argument(
        "--bs",
        type=int,
        help="batch size"
    )

    parser.add_argument(
        "--eppt",
        type=int,
        help="number of pretrain epochs, where eppt+epft = total epochs"
    )

    parser.add_argument(
        "--epft",
        type=int,
        help="number of finetune epochs, where eppt+epft = total epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate"
    )

    parser.add_argument(
        "--wd",
        type=float,
        help="weight decay, l2 reg"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="seed"
    )

    parser.add_argument(
        "--jname",
        type=str,
        help="cluster's job name, otherwise local"
    )



    args = parser.parse_args()


    # start training neural network!!
    nn = NNTrain(
        dataset_name=args.dataset,
        ssl_method=args.ssl,
        encoder_name=args.encoder,
        model_name=args.model,
        optimizer_name=args.opt, 
        batch_size=args.bs, 
        epochs_pt=args.eppt, 
        epochs_ft=args.epft, 
        learning_rate=args.lr, 
        weight_decay=args.wd, 
        seed=args.seed,
        jname=args.jname,
        save_toggle=True
    )
