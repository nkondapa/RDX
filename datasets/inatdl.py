# from https://github.com/lvyilin/pytorch-fgvc-dataset/

import os
import pandas as pd
import warnings
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, extract_archive
import numpy as np
import torch
import json

class INatDL(torch.utils.data.Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """

    def __init__(self, root, split=None, transform=None, target_transform=None, class_list=None):
        super(INatDL, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.class_list = class_list
        self.loader = default_loader
        with open(os.path.join(root, 'data.json'), 'r') as f:
            data = json.load(f)

        self.paths = data['file_paths']
        self.labels = data['class_labels']
        self.class_names = data['class_names']
        self.num_classes = len(self.class_names)
        if split is not None:
            self.train_indices = data['train_indices']
            self.test_indices = data['test_indices']
            if split == 'train':
                self.paths = [self.paths[i] for i in self.train_indices]
                self.labels = [self.labels[i] for i in self.train_indices]
            elif split == 'test':
                self.paths = [self.paths[i] for i in self.test_indices]
                self.labels = [self.labels[i] for i in self.test_indices]
            else:
                raise ValueError("split should be either 'train' or 'test'")

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)

        if class_list is not None:
            mask = np.isin(self.labels, class_list)
            self.labels = self.labels[mask]
            self.paths = self.paths[mask]

        print([self.class_names[i] for i in np.unique(self.labels)])
        print(len(self.paths), "images in the dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])
        target = self.labels[idx]
        img = self.loader(path)

        if self.transform is not None:
            if self.modified_classes is not None and (target in self.modified_classes or "all" in self.modified_classes):
                img = self.modified_transform(img)
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return dict(input=img, target=target, path=path)



if __name__ == '__main__':
    train_dataset = INatDL(root='/home/nkondapa/PycharmProjects/inat_downloader/results', split='train', transform=None, target_transform=None)
    test_dataset = INatDL(root='/home/nkondapa/PycharmProjects/inat_downloader/results', split='test', transform=None, target_transform=None)
    dataset = INatDL(root='/home/nkondapa/PycharmProjects/inat_downloader/results', split=None, transform=None, target_transform=None)