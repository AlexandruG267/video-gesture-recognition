import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

import torchvision.transforms as transforms

# for visualizing shadowy images
import matplotlib.pyplot as plt
import numpy as np
import random

# my root folder: change at will
root = "data/videos/small-20bn-jester-v1"

# Shadowy image (mean pixel value across Time dimension)
class JesterMeanBaselineDataset(Dataset):
    def __init__(self, data_root, annotation_file, transform=None, text_label_dict=None, trim_percent=0.3):
        self.data_root = data_root
        self.transform = transform
        self.trim_percent = trim_percent  # effectively trims the images by 2 * trim_percent. This is done to 
        # keep mostly relevant frames from the image, as usually the first trim_percent frames is the 
        # subject starring at the camera, motionless, and so are the last trim_percent frames, making
        # the output image noisy, or motionless
        
        # load CSV data
        df = pd.read_csv(annotation_file, sep=';', header=None, names=['video_id', 'label'])
        self.video_ids = df['video_id'].astype(str).tolist()
        raw_labels = df['label'].tolist()

        # id_to_label_map for future lookup of predictions (so we can see what the model predicts in language. not numbers)
        self.id_to_label_map = pd.Series(df.label.values, index=df.video_id).to_dict()

        if text_label_dict is not None:
            self.class_to_idx = text_label_dict
        else:
            # creates the gesture: numeric_label map, from the gestures in train. This will be important for Validation later
            unique_labels = sorted(list(set(raw_labels)))
            self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
            
        self.labels = [self.class_to_idx[l] for l in raw_labels]

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = self.labels[idx]
        video_dir = os.path.join(self.data_root, video_id)

        try:
            frame_names = sorted([x for x in os.listdir(video_dir) if x.endswith('.jpg')])
            # debugging: seeing how many frames there are at the beginning
            # print(f"Video {video_id}: First={frame_names[0]}, Last={frame_names[-1]}")
        except FileNotFoundError:
            print("missed some image")
            return torch.zeros(1), label

        total_frames = len(frame_names)
        
        # calculate how many frames to drop from each side
        cut_amount = int(total_frames * self.trim_percent)
        
        # it keeps everything if cut is 0.0
        if cut_amount > 0:
            # revert to keeping only the middle frame if we cut too much (trim_percent >= 0.5)
            if (total_frames - (2 * cut_amount)) <= 0:
                mid = total_frames // 2
                frame_names = [frame_names[mid]]
            else:
                # trim it up
                frame_names = frame_names[cut_amount : -cut_amount]

        self.frames_available = len(frame_names)

        # debugging: seeing how many images are left, from how many there were (previous print)
        # print(f"Video {video_id}: First={frame_names[0]}, Last={frame_names[-1]}")

        tensors = []
        for frame_name in frame_names:
            img_path = os.path.join(video_dir, frame_name)
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            tensors.append(img)

        # stack all frames. so  shape (32, 3, H, W)
        stacked_video = torch.stack(tensors)
    
        # getting the mean along "Time", sesulting in a shape of (3, H, W)
        mean_image = torch.mean(stacked_video, dim=0)
        
        return mean_image, label

# function for visualization (will help for the report)
def show_random_baseline_image(dataset):
    """
    Picks a random sample from the dataset, converts the tensor back to a 
    viewable image, and displays it with its label.
    """
    idx = random.randint(0, len(dataset) - 1)
    
    img_tensor, label_idx = dataset[idx]
    
    # Matplotlib expects images in format (Height, Width, Channels)
    # so we permute dimensions: (3, H, W) -> (H, W, 3)
    img_view = img_tensor.permute(1, 2, 0).numpy()
    
    # We invert the class_to_idx dictionary to get the text back
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label_text = idx_to_class.get(label_idx, "Unknown")

    plt.figure(figsize=(4, 4))
    plt.imshow(img_view)
    plt.title(f"Label: {label_text} (ID: {label_idx})\n'Shadowy' Mean Image")
    plt.axis('off')
    plt.show()

    

# loading and testing script
use_small_train = True

if use_small_train:
    train_path_of_choice = "jester-v1-small-train.csv"
    data_download_path = 'downloaded_data_small/small-20bn-jester-v1/'
else:
    train_path_of_choice = "jester-v1-train.csv"
    data_download_path = 'something_else_entirely'

# should be the same for all of us
valid_path = "data/labels/jester-v1-validation.csv"

transform = transforms.Compose([
    transforms.Resize((100, 150)),
    transforms.ToTensor()
])

baseline_data_train = JesterMeanBaselineDataset(
    data_root=root,
    annotation_file=train_path_of_choice,
    transform=transform
)

label_map = baseline_data_train.class_to_idx

baseline_data_valid = JesterMeanBaselineDataset(
    data_root=root,
    annotation_file=valid_path,
    transform=transform,
    text_label_dict=label_map
)

# print(baseline_data_valid[8])

# seeing the Shadowy image
show_random_baseline_image(baseline_data_valid)