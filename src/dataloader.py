from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

class_to_idx = {
    cls: idx
    for idx, cls in enumerate(CLASS_NAMES)
}


class SkinDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_folder,
        transform=None
    ):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        img_name = row["image_id"] + ".jpg"
        img_path = os.path.join(self.image_folder,img_name)
        image = Image.open(img_path).convert("RGB")
        label = class_to_idx[row["dx"]]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)