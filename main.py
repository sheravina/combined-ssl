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
        type=str,
        help="select a ssl method (simclr, moco, simsiam, jigsaw, vae, simmim)",
    )  # ignored if model is supervised

    args = parser.parse_args()
    
    # start training neural network!! 
    # print(args)


    try:
        # al.run()
        # parser.print_help()
        nn = NNTrain(
                dataset_name=args.dataset,
                ssl_method=args.ssl,
                encoder_name=args.encoder,
                model_name=args.model
            )
        # pprint(nn.__dict__)
    except Exception as e:
        # file_path = al.data_logger.log_primary_results(al.test_loss_list, al.best_dropout_rate_list, al.best_l2_reg_list, al.test_accuracy_list)
        # al.log_params()
        # al.visualizer.plot_results(file_path)
        print("An error occurred! ")
        print(e)
