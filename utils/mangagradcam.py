#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
from torch import nn, optim
import cv2


class MangaGradCAM(nn.Module):
    """Class for Wrapping GradCAM Functions to the ManGanda model"""
    def __init__(self, ManGanda):
        super(MangaGradCAM, self).__init__()
        
        # get the pretrained ManGanda network
        self.model = ManGanda.model
        for p in self.model.parameters():
            p.requires_grad = True
            
        # disect the network to access its last convolutional layer
        self.features_conv = self.model[0]
        
        # get the regressor of the ManGanda
        self.regressor = self.model[1]
        
        # placeholder for the activations and gradients
        self.activations = None
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
    
        self.activations = x
        h = x.register_hook(self.activations_hook)
        
        out = self.regressor(x)
        
        return out
    
    def get_gradient(self):
        return self.gradients

    
def get_cam(cam_model, input_tensor):
    """Get the CAM Saliencies for the input tensor"""
    out = cam_model(input_tensor)
    activations = cam_model.activations
    out[0].backward()
    
    # pull the gradients out of the model
    gradients = cam_model.get_gradient()
    
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    saliency = torch.mean(activations, dim=1).squeeze().detach()
    saliency = np.maximum(saliency, 0)

    # normalize the heatmap
    saliency /= torch.max(saliency)

    # resize
    saliency = cv2.resize(saliency.detach().numpy(),(224,224))

    return saliency