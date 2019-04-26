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
        
def IOU(alpha_gt: torch.Tensor, alpha: torch.Tensor, threshold=0.2):
    alpha_gt = alpha_gt > threshold
    alpha = alpha > threshold
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
    
    iou_threshold = [0.2, 0.4, 0.6, 0.8, 0.9]
    for epoch in range(start_epoch, start_epoch+num_epochs):
        loss_ = 0.
        iou_ = [0.]*len(iou_threshold)
        
        for i, batch in enumerate(train_data):
            feat, img, alpha_gt = batch
            feat, img, alpha_gt = feat.to(device), img.to(device), alpha_gt.to(device)
            
            alpha = model(feat, img)
            loss = criterion(alpha, alpha_gt)
            iou = [IOU(alpha_gt, alpha, t) for t in iou_threshold]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_ += loss.item()
            for j in range(len(iou_threshold)):
                iou_[j] += iou[j]
            print('loss:', loss_ / (1+i))
            
            trainlog.add_scalar('L1', loss.item())
            for j in range(len(iou_threshold)):
                trainlog.add_scalar('IOU_'+str(iou_threshold[j]), iou[j])
            trainlog.add_image('alpha_gt', vutils.make_grid(alpha_gt, normalize=False, nrow=3))
            trainlog.add_image('alpha', vutils.make_grid(alpha, normalize=False, nrow=3))
            trainlog.add_image('alpha_diff', vutils.make_grid(alpha-alpha_gt, normalize=False, nrow=3))
            if (i+1) % 20 == 0:
                for var_name, value in model.named_parameters():
                    # 万一网络中有不在本次训练iterate的参数
                    if not hasattr(value.grad, 'data'):
                        continue
                    var_name = var_name.replace('.', '/')
                    trainlog.add_histogram(var_name, value.data.cpu().numpy())
                    trainlog.add_histogram(var_name+'/grad', value.grad.data.cpu().numpy())
            
            trainlog.next_step()
            
        print('Epoch %d, avg loss: %.3f'%(epoch, loss_ / (i+1)))
        trainlog.save_model(model, epoch)
        for j in range(len(iou_threshold)):
            trainlog.add_scalar('avg_IOU_'+str(iou_threshold[j]), iou_[j]/(i+1), epoch)
              
        model.eval()
        loss_valid_ = 0.
        iou_valid_2, iou_valid_4, iou_valid_6, iou_valid_8, iou_valid_9 = 0., 0., 0., 0., 0.
        for j, v_data in enumerate(valid_data):
            feat, img, alpha_gt = v_data
            feat, img, alpha_gt = feat.to(device), img.to(device), alpha_gt.to(device)
            alpha = model(feat, img)
            loss = criterion(alpha, alpha_gt)
            iou_2 = IOU(alpha_gt, alpha, 0.2)
            iou_4 = IOU(alpha_gt, alpha, 0.4)
            iou_6 = IOU(alpha_gt, alpha, 0.6)
            iou_8 = IOU(alpha_gt, alpha, 0.8)
            iou_9 = IOU(alpha_gt, alpha, 0.9)
              
            loss_valid_ += loss.item()
            iou_valid_2 += iou_2
            iou_valid_4 += iou_4
            iou_valid_6 += iou_6
            iou_valid_8 += iou_8
            iou_valid_9 += iou_9
            print('loss: %.3f'%(loss_valid_/(j+1)))
        trainlog.add_scalar('loss_val', loss_valid_/(j+1), epoch)
        trainlog.add_scalar('iou_val.2', iou_valid_2/(j+1), epoch)
        trainlog.add_scalar('iou_val.4', iou_valid_4/(j+1), epoch)
        trainlog.add_scalar('iou_val.6', iou_valid_6/(j+1), epoch)
        trainlog.add_scalar('iou_val.8', iou_valid_8/(j+1), epoch)
        trainlog.add_scalar('iou_val.9', iou_valid_9/(j+1), epoch)
        print('Epoch %d validation, avg_loss: %.3f'%(epoch, loss_valid_/(j+1)))
        model.train()
        
if __name__ == '__main__':
    main()