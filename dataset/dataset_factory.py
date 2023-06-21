import torch
import os
from torchvision import transforms as T

from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder


class DatasetFactory:

    def __init__(self, dataset_name, split, data_path):
        self.split = split
        self.transform = T.Compose(
            [
                T.ToTensor()
            ]
        )
        self.dataset_name = dataset_name
        self.data_path = data_path

    def get_dataset(self):
        is_train = False
        if self.split == 'train':
            is_train = True

        if self.dataset_name == 'cifar-10':
            selected_dataset = CIFAR10(root=self.data_path, train=is_train, transform=self.transform, download=True)
        elif self.dataset_name == 'cifar-100':
            selected_dataset = CIFAR100(root=self.data_path, train=is_train, transform=self.transform, download=True)
        elif self.dataset_name == 'svhn':
            selected_dataset = SVHN(root=self.data_path, split=self.split, transform=self.transform,
                                    download=True)
        elif self.dataset_name == 'imagenet-100':
            selected_dataset = ImageFolder(root=os.path.join(self.data_path, self.split),
                                           transform=self.transform)
        else:
            raise ValueError

        sorted_dataset = sorted(selected_dataset, key=lambda x: x[1])
        #sorted_dataset = torch.stack([data for data, _ in sorted_dataset])
        return sorted_dataset
