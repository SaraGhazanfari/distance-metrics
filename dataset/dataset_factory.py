import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms as T

from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder


class DatasetFactory:

    def __init__(self, dataset_name, split, data_path, batch_size, num_workers=1, shuffle=False, drop_last=False,
                 pin_memory=True, has_normalize=True):
        self.split = split
        self.transform = T.Compose(
            [
                T.ToTensor()
            ]
        )
        self.imagenet_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor()
            ]
        )

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.has_normalize = has_normalize
        self.num_workers = num_workers

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
                                           transform=self.imagenet_transform)
        else:
            raise ValueError

        sorted_dataset = sorted(selected_dataset, key=lambda x: x[1])
        sorted_dataset = [data for data, _ in sorted_dataset]

        dataloader = DataLoader(
            sorted_dataset,
            batch_size=self.batch_size,  # may need to reduce this depending on your GPU
            num_workers=self.num_workers,  # may need to reduce this depending on your num of CPUs and RAM
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return sorted_dataset, dataloader
