import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(128, 64, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU()) # depth 128-> 64
        self.conv_2 = nn.Sequential(nn.Conv2d(64, 32, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU()) # depth 64-> 32
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 16, 1, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()) # depth 32-> 16
        self.conv_4 = nn.Sequential(nn.Conv2d(16, 8, 1, 1),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU()) # depth 16-> 8
        
        # extract low level features from origin image
        self.low_level_1 = nn.Sequential(nn.Conv2d(3, 16, 5, 1, 2), 
                                         nn.BatchNorm2d(16))
        self.low_level_2 = nn.Sequential(nn.Conv2d(16, 8, 3, 1, 2), 
                                         nn.BatchNorm2d(8),
                                         nn.ReLU())
        self.low_level_3 = nn.Sequential(nn.Conv2d(8, 4, 3, 1, 2), 
                                         nn.BatchNorm2d(4),
                                         nn.ReLU())
        
        # after concat with origin image, depth becomes 12
        self.conv_5 = nn.Sequential(nn.Conv2d(12, 6, 1, 1),
                                   nn.BatchNorm2d(6),
                                   nn.ReLU())
        self.conv_6 = nn.Sequential(nn.Conv2d(6, 3, 1, 1),
                                   nn.BatchNorm2d(3),
                                   nn.ReLU())
        self.conv_7 = nn.Sequential(nn.Conv2d(3, 1, 1, 1),
                                   nn.ReLU())
        
    def forward(self, feat, img):
        feat = self.conv_1(feat)
        feat = self.conv_2(feat)
        feat = self.conv_3(feat)
        feat = self.conv_4(feat)
        
        img = self.low_level_1(img)
        img = self.low_level_2(img)
        img = self.low_level_3(img)
        
        mix = torch.cat((feat, img), 1)
        mix = self.conv_5(mix)
        mix = self.conv_6(mix)
        mix = self.conv_7(mix)
        
        return mix