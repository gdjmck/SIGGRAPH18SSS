from tensorboardX import SummaryWriter
import numpy as np
import argparse
import math
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset
import torch_net

num_epochs = 200

def main():
    device = torch.device('cpu')
    model = torch_net.FCN()
    model.to(device)
    
    train_data = DataLoader(dataset.InputData(data_folder='./samples'),
                            batch_size=6,
                            drop_last=True,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=True)
    model.train()
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(),
                           lr = lr, betas=(0.9, 0.999),
                           weight_decay=0.0005)
    criterion = nn.L1Loss()
    
    for epoch in range(num_epochs):
        loss_ = 0
        
        for i, batch in enumerate(train_data):
            feat, img, alpha_gt = batch
            feat, img, alpha_gt = feat.to(device), img.to(device), alpha_gt.to(device)
            
            alpha = model(feat, img)
            loss = criterion(alpha, alpha_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_ += loss.item()
            print(loss_ / (1+i))
        
if __name__ == '__main__':
    main()