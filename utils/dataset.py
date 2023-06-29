#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


# Custom Manga Dataset
class MangaDataset(Dataset):
    """Custom Dataset for mangas"""
    def __init__(self, w_val=True):
        """Get annotations of mangas and split for training"""
        self.annotations = pd.read_csv('saves/annotations.csv')
        trainval, test = train_test_split(self.annotations.index.tolist(),
                                          stratify=self.annotations.rating,
                                          test_size=0.1,
                                          random_state=143)
        train, val = train_test_split(
            trainval,
            stratify=self.annotations.loc[trainval].rating,
            test_size=1/6,
            random_state=143
        )

        # Size of datasets
        self.sizes = {
            'train': len(train),
            'val': len(val),
            'test': len(test),
        }

        self.train = Subset(self, train)
        self.val = Subset(self, val)
        self.test = Subset(self, test)
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """Get manga item and apply preset transformations"""
        path = self._get_manga_path(index)
        label = self._get_manga_label(index)
        title = self._get_manga_title(index)
        
        image = transforms.ToTensor()(Image.open(path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        dim = min(torch.tensor(image.shape[1:])).item()

        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.CenterCrop(dim),
            transforms.Resize(224, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        image = self.transform(image)
        
        return image, label, path, title

    def _get_manga_path(self, index):
        return self.annotations.iloc[index, 0]

    def _get_manga_label(self, index):
        return self.annotations.iloc[index, 1]
    
    def _get_manga_title(self, index):
        return self.annotations.iloc[index, 2]