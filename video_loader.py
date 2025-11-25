import os
import csv
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

data_root = 'data'
# labels_dir = 'labels'
# videos_dir = 'videos/small-20bn-jester-v1'


class JesterVideos(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the JesterVideos dataset with the root directory for the images,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for the JesterVideos images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        self.label_dict = label_dict if label_dict is not None else {}

        with open(os.path.join(self.root_dir, self.split + '.txt')) as r:
            lines = r.readlines()
            for line in lines:
                line = line.split()
                # print(line)
                self.filenames.append(line[0])
                if split == 'test':
                    label = line[0]
                else:
                    label = int(line[1])
                self.labels.append(label)
                if split == 'train':
                    # print(line[0].split("/"))
                    # In this one, os.sep doesn't work ↓
                    # text_label = line[0].split(os.sep)[2]
                    text_label = line[0].split("/")[2]
                    self.label_dict[label] = text_label


    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a single image and its corresponding label when given an index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        if self.transform is not None:
            image = self.transform(
                Image.open(os.path.join(self.root_dir, "images", self.filenames[idx])))
        else:
                image = Image.open(os.path.join(self.root_dir, "images", self.filenames[idx]))
        label = self.labels[idx]
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        # transforms.Resize((128,128)),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    return train_transform, test_transform


def get_dataset(transform, type):
    match type:
        case "train":
            dataset = JesterVideos(data_root,
                                   split='train',
                                   transform=transform)
        case "val":
            dataset = JesterVideos(data_root,
                                   split='val',
                                   transform=transform,
                                   label_dict=miniplaces_train.label_dict)
        case "test":
            dataset = JesterVideos(data_root,
                                   split='test',
                                   transform=transform)
        case _:
            raise ValueError("Type must be train, val or test")
    return dataset

def get_loader(dataset, type, batch_size, num_workers):
    match type:
        case "train":
            loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=True)
        case "val":
            loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=False)
        case "test":
            loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=False)
    
    return loader