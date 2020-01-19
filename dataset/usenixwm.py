from __future__ import print_function

import os
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

from dataset.cifar100 import get_cifar100_train_dataset

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}
std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = 'data/watermark/USENIX'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class UsenixWM(Dataset):
    """CIFAR100Instance Dataset.
    """
    def __init__(self, transform=None):
        """
        @:param train_data (dataset, optional): Dataset to mix watermark with
        @:param transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = get_data_folder()

        self.items = os.listdir(self.root_dir)
        if self.__len__() == 0:
            raise FileNotFoundError("Could not find watermark dataset location!")

        self.transform = transform

        num_classes = 100
        self.targets = np.random.randint(num_classes, size=self.__len__())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        img_name = os.path.join(os.getcwd(), os.path.join(self.root_dir, self.items[index]))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        # input - target - index
        return img, self.targets[index]


def get_usenixwm_cifar100_dataloader(batch_size=128, n_train=1000, num_workers=8):
    """
    Gets a mix of the usenix watermark and cifar data

    """
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = get_cifar100_train_dataset(n_train=n_train)
    wm_set = UsenixWM(transform=train_transform)

    all_data = ConcatDataset([train_set, wm_set])

    train_loader = DataLoader(all_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    return train_loader

def get_usenixwm_dataloader(batch_size=128, num_workers=8):
    """
    :param train_loader (object, optional): Samples from the training dataset
    """
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = UsenixWM(transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    return train_loader
