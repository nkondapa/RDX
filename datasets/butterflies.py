import os
import torch
from PIL import Image
import json
import numpy as np

class ButterfliesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split=None, transform=None):
        self.root = root
        self.transform = transform
        with open(os.path.join(self.root, 'class_names.json'), 'r') as f:
            self.classes = json.load(f)

        self.paths = []
        self.labels = []
        for ci, class_name in enumerate(self.classes):
            folder_name = os.path.join('images', class_name)
            class_path = os.path.join(self.root, folder_name)
            image_paths = os.listdir(class_path)
            for image_path in image_paths:
                self.paths.append(os.path.join(folder_name, image_path))
                self.labels.append(ci)

        with open(os.path.join(self.root, 'train_test_split.json'), 'r') as f:
            self.split_dict = json.load(f)

        self.split = split

        if self.split is not None:
            split_inds = self.split_dict[split]
            new_paths = []
            new_labels = []
            for i in range(len(self.paths)):
                if i in split_inds:
                    new_paths.append(self.paths[i])
                    new_labels.append(self.labels[i])
            self.paths = new_paths
            self.labels = new_labels

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return dict(input=image, target=self.labels[idx])


if __name__ == '__main__':
    for split in ['train', 'test', None]:
        dataset = ButterfliesDataset('/home/nkondapa/Datasets/butterflies', split=split, transform=None)
        print(len(dataset))
        print(dataset[0])




