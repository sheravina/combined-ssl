# from torchvision import transforms
# import random
# from PIL import ImageFilter
# from utils.constants import *

# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x

# #https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

# def normval(dataset):
#     if dataset == CIFAR10_DATASET:
#         norm_mean = (0.4914, 0.4822, 0.4465)
#         norm_std = (0.2023, 0.1994, 0.2010)
#     elif dataset == CIFAR100_DATASET:
#         norm_mean = (0.5071, 0.4865, 0.4409)
#         norm_std = (0.2673, 0.2564, 0.2762)
#     elif dataset == SVHN_DATASET:
#         norm_mean =  (0.4377, 0.4438, 0.4728)
#         norm_std = (0.1980, 0.2010, 0.1970)
#     return norm_mean, norm_std

# norm_mean, norm_std = normval(_____)


# #CIFAR-10
# # norm_mean = (0.4914, 0.4822, 0.4465)
# # norm_std = (0.2023, 0.1994, 0.2010)

# #CIFAR-100
# #norm_mean = (0.5071, 0.4865, 0.4409)
# #norm_std = (0.2673, 0.2564, 0.2762)

# #SVHN
# #norm_mean = (0.4377, 0.4438, 0.4728)
# #norm_std = (0.1980, 0.2010, 0.1970)

# # base_transformation = transforms.ToTensor()

# # train_transformation = transforms.Compose(
# #     [
# #         transforms.RandomCrop(32, padding=4),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(norm_mean, norm_std)
# #     ]
# # )

# # test_transformation = transforms.Compose(
# #     [
# #         transforms.ToTensor(),
# #         transforms.Normalize(norm_mean, norm_std)
# #     ]
# # )

# # simclr_transformation = transforms.Compose(
# #     [   
# #         # transforms.RandomCrop(32, padding=4),
# #         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
# #         transforms.RandomHorizontalFlip(p=0.5),
# #         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
# #         transforms.RandomGrayscale(p=0.2),
# #         # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
# #         transforms.ToTensor(),
# #         transforms.Normalize(norm_mean, norm_std),
# #     ]
# # )

# # inet_transform = transforms.Compose([
# #     transforms.Resize((299, 299)),  
# #     *train_transformation.transforms
# # ])

# # inet_simclr_transform = transforms.Compose([
# #     transforms.Resize((299, 299)), 
# #     transforms.RandomResizedCrop(size=299, scale=(0.2, 1.0)),
# #     transforms.RandomHorizontalFlip(p=0.5),
# #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
# #     transforms.RandomGrayscale(p=0.2),
# #     transforms.ToTensor(),
# #     transforms.Normalize(norm_mean, norm_std),
# # ])