import os
import sys

import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np


# data_path = r'/lxw/mimic/output/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'


class CT2(torch.utils.data.Dataset):

    def __init__(self,
                 data_df, transform, data_path
                 ):
        self.images = []
        self.labels = []
        self.races = []
        self.transform = transform
        self.data_path = data_path

        for index, row in data_df.iterrows():
            self.images.append(row['path'])
            self.labels.append(row['label'])
            self.races.append(row['race'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.data_path, self.images[item]))
        label = self.labels[item]
        label_numpy = np.array(int(label))
        label_tensor = torch.from_numpy(label_numpy).to(torch.int64)
        race_label = self.races[item]
        race_numpy = np.array(int(race_label))
        race_tensor = torch.from_numpy(race_numpy).to(torch.int64)
        return item, self.transform(img), label_tensor, race_tensor


train_transforms2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将输入的灰度图像转换为彩色图像

    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop((HEIGHT, WIDTH), scale=(0.9, 1.1)),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将输入的灰度图像转换为彩色图像
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_transforms = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),  # 将输入的灰度图像转换为彩色图像

    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop((HEIGHT, WIDTH), scale=(0.9, 1.1)),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

val_transforms = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),  # 将输入的灰度图像转换为彩色图像
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])


def get_dataset(dataset, data_df, data_path, trans_name):
    if dataset == 'CT_L':
        transform = train_transforms
        if trans_name != 'train':
            transform = val_transforms
        dataset = CT2(data_df=data_df, transform=transform, data_path=data_path)
    elif dataset == 'CT':
        transform = train_transforms2
        if trans_name != 'train':
            transform = val_transforms2
        dataset = CT2(data_df=data_df, transform=transform, data_path=data_path)
    else:
        print('dataset error.')
        sys.exit(0)
    return dataset
