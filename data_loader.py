import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import pandas as pd
import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi


class GTSRB(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[["Path", "ClassId"]]
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
    def __init__(self, _batch_size, download=False):
        super(Dataset, self).__init__()
        code_dir = os.getcwd()
        img_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
        if download:
            Dataset.download_data(img_dir)
        train_file = os.path.join(img_dir, "Train.csv")
        test_file = os.path.join(img_dir, "Test.csv")
        self.batch_size = _batch_size
        self.train_data = GTSRB(
            img_dir=img_dir,
            annotations_file=train_file,
            transform=transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        self.test_data = GTSRB(
            img_dir=img_dir,
            annotations_file=test_file,
            transform=transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False)

    @staticmethod
    def download_data(data_dir):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        else:
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        os.chdir(data_dir)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign',
            path=data_dir)
        return_code = os.system("unzip gtsrb-german-traffic-sign.zip")
        if return_code != 0:
            print("Could not unzip file")
            exit(1)
        print("Download complete")
