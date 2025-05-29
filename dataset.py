import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class XBDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['filename']))
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_str = row['label']
        label = torch.tensor(self.label_mapping[label_str], dtype=torch.long)

        return image, label
