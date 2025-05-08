from torchvision import transforms

norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)

base_transformation = transforms.ToTensor()

basenorm_transformation = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

basenorm_jp_transformation = transforms.Compose( #with chances of grayscale
    [
        transforms.ToTensor(),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

simclr_transformation = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

inet_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 🔥 Required for InceptionV3!
    *basenorm_transformation.transforms
])

inet_simclr_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 🔥 Required for InceptionV3!
    transforms.RandomResizedCrop(size=299, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])