import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2

class HairDataset(data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.counter = self.df.shape[0]
        self.image_agg = self.df['image'].values
        self.labels = self.df['type'].values


    def __len__(self):
        return self.counter

    def __getitem__(self, idx):
        image_path = self.image_agg[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_path, image, label


if __name__ == '__main__':
    hair_dataset = HairDataset('./data/hair_class.csv')
    image_name, _, label = hair_dataset.__getitem__(1)
    print(image_name, '\t', label)