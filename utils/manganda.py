#!/usr/bin/env python
# coding: utf-8


from tqdm.notebook import tqdm, trange
import torch, time, copy
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary


def train_model(ManGanda, criterion, optimizer, num_epochs=125):
    since = time.time()

    best_model_wts = copy.deepcopy(ManGanda.model.state_dict())
    best_loss = 1_000_000
    best_epoch = 0
    
    for epoch in trange(num_epochs):
        # Check if model no longer improves
        if best_epoch >= 10 + epoch:
            break
        
        log = f'Epoch {epoch+1:2d}/{num_epochs}  |'
        # Each epoch has a training and validation phase
        for stage in ['train', 'val']:
            if stage == 'train':
                ManGanda.model.train()  # Set model to training mode
                epoch_loader = ManGanda.dataloader.train
            else:
                ManGanda.model.eval()   # Set model to evaluate mode
                epoch_loader = ManGanda.dataloader.val
                
            running_loss = 0.0

            # Iterate over data.
            for X, y in tqdm(epoch_loader):
                X = X.to(ManGanda.device)
                y = y.float().to(ManGanda.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(stage == 'train'):
                    out = ManGanda.model(X)
                    loss = criterion(out, y)
                    
                    # backward + optimize only if in training phase
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * X.size(0)
           
            epoch_loss = running_loss / ManGanda.dataset.sizes[stage]
            
            if stage == 'val':
                stage = 'validation'
            log += (f'  {stage.title()} Loss: {epoch_loss:.4f}  |')

            # deep copy the model
            if stage == 'validation' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(ManGanda.model.state_dict())
                torch.save(best_model_wts, f'saves/MangaModel.pth')
                best_epoch = epoch
                
        print(log)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m'
          f'{time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_loss:4f}')

    # load best model weights
    ManGanda.model.load_state_dict(best_model_wts)
    
    return ManGanda.model


class MangaModel(nn.Module):
    """Regression Model for Rating Mangas"""
    def __init__(self, dataset, dataloader, num_epochs=500, save_file='saves/MangaModel.pth'):
        super(MangaModel, self).__init__()
        """Initialize with the trained parameters, else retrain from scratch
        """        
        # Get the device for loading the model
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        
        # Embed the dataset and dataloader
        self.dataset = dataset
        self.dataloader = dataloader
        
        self.model = torch.hub.load('RF5/danbooru-pretrained', 'resnet18',
                                    pretrained=False)

        # Load the pretrained weights
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet18-3f77756f.pth',
            map_location=self.device.type
        )
        state_dict = {key.replace("module.", ""): value 
                      for key, value in checkpoint.items()}

        self.model.load_state_dict(state_dict)

        # Freezing the weights of the pretrained model
        for param in self.model[0].parameters():
            param.requires_grad = False
        self.model[1][8] = nn.Linear(512, 1)
        self.model[1].append(nn.Threshold(0, 10))
            
        self.model.to(self.device)
        summary(self.model, (3, 224, 224))
        
        # Set the loss function for Regression
        self.criterion = nn.MSELoss()

        # Only the parameters of the regressor are being optimized
        self.optimizer = optim.Adam(self.model[1].parameters(), lr=0.001)
        
        # Load the retrained model, else, train
        try:
            self.model.load_state_dict(torch.load(save_file))
        except:
            self.model = train_model(self,
                                     self.criterion,
                                     self.optimizer,
                                     num_epochs=num_epochs)
    def forward(self, X):
        return self.model(X)