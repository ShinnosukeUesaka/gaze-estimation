import torch
import torch.nn as nn
from torchvision import transforms
from modules import resnet50
from modules.function import ReverseLayerF
import torch.nn.functional as F
import numpy as np

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl


import utils

class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze
    
class GazeModel(pl.LightningModule):
    def __init__(self, lr, lr_patience, lr_decay_factor):
        super().__init__()
        
        self.gaze_network = resnet50(pretrained=True)
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )
        
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay_factor = lr_decay_factor
        
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=["loss_fx"])
    
    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)
        return gaze
    
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, target_gaze = batch
        pred_gaze= self(image)
        loss = F.l1_loss(pred_gaze, target_gaze)
        angular_error = np.mean(utils.angular_error(pred_gaze.cpu().data.numpy(), target_gaze.cpu().data.numpy()))
        values = {"test_loss": loss, "test_angular_error": angular_error}  # add more items if needed
        self.log_dict(values,  prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, target_gaze = batch
        pred_gaze= self(image)
        loss = F.l1_loss(pred_gaze, target_gaze)
        angular_error = np.mean(utils.angular_error(pred_gaze.cpu().data.numpy(), target_gaze.cpu().data.numpy()))
        values = {"val_loss": loss, "val_angular_error": angular_error}  # add more items if needed
        self.log_dict(values,  prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, target_gaze = batch
        pred_gaze= self(image)
        loss = F.l1_loss(pred_gaze, target_gaze)
        angular_error = np.mean(utils.angular_error(pred_gaze.cpu().data.numpy(), target_gaze.cpu().data.numpy()))
        values = {"train_loss": loss, "train_angular_error": angular_error}  # add more items if needed
        self.log_dict(values,  prog_bar=True,)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = StepLR(optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)
        return [optimizer], [scheduler]