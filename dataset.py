import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2

class HairDataset(data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.counter = self.df.shape[0]
        self.image_agg = self.df['image'].values
        self.labels = self.df['type'].values
        self.transform = transform

    def __len__(self):
        return self.counter

    def convert_label(self, label):
        bitmap = np.zeros(5,dtype=np.float32)
        bitmap[label-1] = 1
        return bitmap

    def __getitem__(self, idx):
        image_path = self.image_agg[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.convert_label(label)
        if self.transform:
            image, label = self.transform(image, label)
        return image_path, image, label


if __name__ == '__main__':
    hair_dataset = HairDataset('./data/hair_class.csv')
    image_name, _, label = hair_dataset.__getitem__(1000)
    print(image_name, '\t', label)