#!/usr/bin/env python
# coding: utf-8


import numpy as np
from utils.pickling import *
import torch
from torch.utils.data import DataLoader


class MangaDataloader():
    """Dataloaders for the MangaDataset"""
    def __init__(self, dataset, batch_size=512):
        # Getting Training Distribution
        try:
            means = load_pkl('means')
            stds = load_pkl('stds')
        except:
            means = []
            stds = []
            for img_t, *_ in dataset.train:
                means += [img_t.mean().item()]
                stds += [img_t.std().item()]
            
            # Save if new
            save_pkl(means, 'means')
            save_pkl(stds, 'stds')

        self._mean_thresh = np.quantile(means, 0.01)
        self._std_thresh = np.quantile(stds, 0.01)

        # Building the DataLoader
        self.train = DataLoader(dataset.train,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=self.collate_fn,
                                num_workers=1)
        self.val = DataLoader(dataset.val,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=self.collate_fn,
                              num_workers=1)
        self.test = DataLoader(dataset.test,
                               batch_size=batch_size,
                               shuffle=False,
                               collate_fn=self.collate_fn,
                               num_workers=1)
    
    def collate_fn(self, batch):
        """Collating function for the dataset"""
        X, Y = [], []

        # Gather in lists, and encode labels as indices
        for x, y, _, _ in batch:

            # Outlier Detection
            if (x.mean().item() >= self._mean_thresh and 
                x.std().item() >= self._std_thresh):

                X += [torch.tensor(x)]
                Y += [torch.tensor(y)]

        # Group the list of tensors into a batched tensor
        X = torch.stack(X)
        Y = torch.stack(Y)

        return X, Y