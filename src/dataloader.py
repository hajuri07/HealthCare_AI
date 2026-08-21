import os
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


class SkinDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = class_to_idx[row["dx"]]
        return image, label
    