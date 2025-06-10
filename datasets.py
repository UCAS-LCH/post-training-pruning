import torch
import torchvision.transforms as transforms
import os
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataset import Dataset


def imagenet_dataloader(data_dir='../data/imagenet', batch_size=64, num_samples=1024):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    test_set = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if num_samples > 0:
        train_set = Subset(
            train_set,
            np.random.choice(len(train_set), num_samples).tolist(),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:    
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_class = 1000

    return train_loader, test_loader, num_class

def cifar10_dataloader(data_dir = '../data', batch_size=128, num_samples=1024):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2021])

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    if num_samples > 0:
        train_set = Subset(
            train_set,
            np.random.choice(len(train_set), num_samples).tolist(),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:    
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 10

    return train_loader, test_loader, num_class

def cifar100_dataloader(data_dir = '../data', batch_size=128, num_samples=1024):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    if num_samples > 0:
        train_set = Subset(
            train_set,
            np.random.choice(len(train_set), num_samples).tolist(),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:    
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_class = 100

    return train_loader, test_loader, num_class



