#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm, trange
from IPython.display import display, HTML
from torch import nn, optim
from torchvision import transforms
from utils.mangagradcam import get_cam


def plot_ratings(dataset, toc):
    """Plot the histogram of ratings"""
    plot = (
        dataset.annotations
        .groupby(['title'])['rating'].mean()
        .apply(lambda x: np.round(x*20) / 20)
    )
    bins = int((plot.max() - plot.min())/0.05) + 1
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.hist(plot, bins=bins, color='k', alpha=0.8)
    sns.despine()
    ax.grid(True, axis='y')
    ax.set_ylabel('Number of Mangas')
    ax.set_xlabel('Ratings')
    toc.add_fig('Distribution of Ratings', width=100)
    
    
def get_baseline(dataset):
    """Print the baseline value to beat"""
    baseline = dataset.annotations.rating.std()

    display(HTML(
        f'<b>An acceptable model performance will be MSE < '
        f'{baseline:0.4f}.</b>'
    ))
    
    return baseline


def eval_model(model, dataloader, baseline):
    """Evaluate model on test set"""
    model.eval()   # Set model to evaluate mode
    mse = 0.0
    mae = nn.L1Loss()
    mae_losses = 0.0
    
    # Iterate over data.
    for X, y in dataloader.test:
        X = X.to(model.device)
        y = y.float().to(model.device)
        out = model(X)
        loss = model.criterion(out, y)
        mae_loss = mae(out, y)
        
        # statistics
        mse += loss.item() * X.size(0)
        mae_losses += mae_loss.item() * X.size(0)
        
    test_loss = mse / model.dataset.sizes['test']
    test_mae_loss = mae_losses / model.dataset.sizes['test']
    
    if test_loss < baseline:
        evaluation = 'ManGanda is performing better than the set baseline!'
    else:
        evaluation = 'ManGanda needs more fine-tuning'
        
    display(HTML(
        '<b>'
        f'Test MSE Loss - {test_loss:.2f}<br>'
        f' Baseline - {baseline:.2f}<br><br>'
        '----------------------------------------------------------------<br>'
        f'{evaluation}'
        '<br><br><br>'
        '****************************************************************<br>'
        'Other Metrics:<br>'
        f'Test MAE Loss - {test_mae_loss:.2f}<br>'
        '</b>'
    ))
    
    
def plot_predictions(model, dataset, toc, n=3):
    """Plot sample images and print prediction"""
    df = dataset.annotations
    mangas = np.random.choice(df.title.unique(), n, replace=False)
    
    for i, manga in enumerate(mangas):
        panels = np.random.choice(df[df['title'] == manga].index,
                                  3,
                                  replace=False)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        sample_data = []
        for j, idx in enumerate(panels):
            data = dataset[idx]
            sample_data.append(data)
            ax[j].imshow(transforms.ToPILImage()(data[0]))
            ax[j].axis('off')

        x, y = model.dataloader.collate_fn(sample_data)
        
        if y.shape[0] < 1:
            print('All sampled images are outliers')
        x = x.to(model.device)
        out = model(x)

        toc.add_fig(f'Sample Prediction # {i+1} - {manga}', width=100)
        display(HTML(
            '<center><b>'
            f'Average Model Prediction - {out.mean().item():.2f}<br>'
            f'   Actual Rating - {y.mean().item():.2f}'
            '</b><center><br><br><br>'
        ))
        
        
def plot_test(model, dataloader, toc):
    """Plot the distribution of actual and test ratings"""
    model.eval()   # Set model to evaluate mode
    ys, outs = [], []

    # Iterate over data.
    for X, y in dataloader.test:
        X = X.to(model.device)
        out = model(X)
        outs.extend(out.view(-1).tolist())
        ys.extend(y.tolist())

    outs = [np.round(x*20) / 20 for x in outs]
    ys = [np.round(x*20) / 20 for x in ys]

    bins_out = int((max(outs) - min(outs))/0.05) + 1
    bins_y = int((max(ys) - min(ys))/0.05) + 1

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.hist(ys, bins=bins_y, color='k', label='True Ratings')
    ax.hist(outs, bins=bins_out, color='gray', alpha=0.8,
            label='Predicted Ratings')

    sns.despine()
    plt.legend()
    ax.grid(True, axis='y')
    ax.set_ylabel('Number')
    ax.set_xlabel('Ratings')
    toc.add_fig('Distribution of Test Ratings (True vs Predicted)', width=100)
    
    
def plot_saliencies(cam_model, model, toc):
    """"""
    df = model.dataset.annotations
    manga = np.random.choice(df.title.unique(), 1, replace=False)[0]
    
    panels = np.random.choice(df[df['title'] == manga].index,
                              3,
                              replace=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sample_data = [model.dataset[idx] for idx in panels]
    
    xs, ys = model.dataloader.collate_fn(sample_data)
    for i, x in enumerate(xs):
        saliency = get_cam(cam_model, x.unsqueeze(0))
        
        ax[i].imshow(transforms.ToPILImage()(x[0]), 'gray')
        ax[i].imshow(saliency, 'jet', alpha=0.60)
        ax[i].axis('off')

    toc.add_fig(f'GradCAM Implementation - {manga}', width=100)