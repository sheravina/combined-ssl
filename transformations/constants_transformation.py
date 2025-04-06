from torchvision import transforms

base_transformation = transforms.ToTensor()

basenorm_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                        ])

simclr_transformation = transforms.Compose([
                            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                            ])
