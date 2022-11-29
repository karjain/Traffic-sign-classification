import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import pandas as pd
import os


class GTSRB(Dataset):
    def __init__(self, annotations_file, img_dir , transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[["Path","ClassId"]]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Dataset:
    def __init__(self, _batch_size):
        super(Dataset, self).__init__()
        img_dir = "/home/ubuntu/pytorch1/"
        train_file = "/home/ubuntu/pytorch1/Train.csv"
        test_file = "/home/ubuntu/pytorch1/Test.csv"
        train_data = GTSRB(img_dir=img_dir, annotations_file=train_file,
                           transform=transforms.Compose(
                               [transforms.Resize((28, 28)), transforms.ConvertImageDtype(torch.float32),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_data = GTSRB(img_dir=img_dir, annotations_file=test_file,
                          transform=transforms.Compose(
                              [transforms.Resize((28, 28)), transforms.ConvertImageDtype(torch.float32),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=_batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=_batch_size, shuffle=False)

