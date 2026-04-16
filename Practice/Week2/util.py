import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def get_base_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_mnist_resnet_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_cifar_resnet_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_mnist_augment_transform():
    '''
    Training transform for MNIST + ResNet18 with light augmentation

    Returns
    -------
    torchvision.transforms.Compose
        ToPILImage → RandomAffine(±10°, translate 10%) → Resize(224)
        → Grayscale(3) → ToTensor → Normalize(ImageNet)
    '''
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_cifar_augment_transform():
    '''
    Training transform for CIFAR-10 + ResNet18 with standard augmentation

    Returns
    -------
    torchvision.transforms.Compose
        ToPILImage → RandomCrop(32, padding=4) → RandomHorizontalFlip
        → Resize(224) → ToTensor → Normalize(ImageNet)

    Notes
    -----
    Reference: He et al., "Deep Residual Learning for Image Recognition", 2015
    '''
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def show_mnist_samples(dataset, n=9):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, n + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(dataset.labels[i])
        plt.axis('off')
        plt.imshow(dataset.img[i], cmap='gray')

    plt.show()


def show_cifar_samples(dataset, label_names, n=9):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, n + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(label_names[dataset.labels[i]].decode('utf-8'))
        plt.axis('off')
        plt.imshow(dataset.img[i].reshape(3, 32, 32).transpose(1, 2, 0))

    plt.show()
