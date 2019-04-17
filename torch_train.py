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
from parse_opt import get_arguments
from torch_logs import TrainLog

num_epochs = 200
        
def IOU(alpha_gt: torch.Tensor, alpha: torch.Tensor):
    alpha_gt = alpha_gt > 0.2
    alpha = alpha > 0.2
    assert alpha_gt.shape == alpha.shape
    
    intersection = (alpha_gt & alpha).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (alpha_gt | alpha).float().sum()         # Will be zzero if both are 0
    # union set is None and return 0
    if union.item() == 0:
        return 0
    
    return (intersection / union).item()

def main():
    # load train args
    args = get_arguments()
    if args.without_gpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('No gpu available')
            return
        
    model = torch_net.FCN()
    model.to(device)
    
    train_data = DataLoader(dataset.InputData(data_folder=args.dataDir),
                            batch_size=args.train_batch,
                            drop_last=True,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True) # 只能将num_workers设为0， 表示只用主线程来load
    valid_data = DataLoader(dataset.InputData(data_folder=args.validDir),
                           batch_size=1,
                           drop_last=False,
                           shuffle=True,
                           num_workers=0)
    
    model.train()
    lr = args.lr
    trainlog = TrainLog(args)
    start_epoch = 1
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr = lr, betas=(0.9, 0.999),
                           weight_decay=0.0005)
    criterion = nn.L1Loss()
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        loss_ = 0.
        iou_ = 0.
        
        for i, batch in enumerate(train_data):
            feat, img, alpha_gt = batch
            feat, img, alpha_gt = feat.to(device), img.to(device), alpha_gt.to(device)
            
            alpha = model(feat, img)
            loss = criterion(alpha, alpha_gt)
            iou = IOU(alpha_gt, alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_ += loss.item()
            iou_ += iou
            print('loss:', loss_ / (1+i), 'iou:', iou_ / (i+1))
            
            trainlog.add_scalar('L1', loss.item())
            trainlog.add_scalar('IOU', iou)
            if (i+1) % 20 == 0:
                for var_name, value in model.named_parameters():
                    # 万一网络中有不在本次训练iterate的参数
                    if not hasattr(value.grad, 'data'):
                        continue
                    var_name = var_name.replace('.', '/')
                    trainlog.add_histogram(var_name, value.data.cpu().numpy())
                    trainlog.add_histogram(var_name+'/grad', value.grad.data.cpu().numpy())
            
            trainlog.next_step()
            
        print('Epoch %d, avg loss: %.3f, avg iou: %.2f'%(epoch, loss_ / (i+1), iou_/(i+1)))
        trainlog.save_model(model, epoch)
              
        model.eval()
        loss_valid_, iou_valid_ = 0., 0.
        for j, v_data in enumerate(valid_data):
            feat, img, alpha_gt = v_data
            feat, img, alpha_gt = feat.to(device), img.to(device), alpha_gt.to(device)
            alpha = model(feat, img)
            loss = criterion(alpha, alpha_gt)
            iou = IOU(alpha_gt, alpha)
              
            loss_valid_ += loss.item()
            iou_valid_ += iou
            print('loss: %.3f, iou: %.2f'%(loss_valid_/(j+1), iou_valid_/(j+1)))
            if j == 4:
                break
        print('Epoch %d validation, avg_loss: %.3f, avg_iou: %.2f'%(epoch, loss_valid_/(j+1), iou_valid_/(j+1)))
        model.train()
        
if __name__ == '__main__':
    main()