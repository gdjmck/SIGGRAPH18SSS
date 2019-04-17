import tensorflow as tf
from parse_opt import get_arguments
from deeplab_resnet import HyperColumn_Deeplabv2, read_data_list
import os
import numpy as np
import torch
import torch.utils.data as data
import main_hyper
from threading import Lock

shape = 250
args = get_arguments()

config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
model = HyperColumn_Deeplabv2(sess, args)
model.load(args.snapshot_dir)
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))    
    return tensor

class InputData(data.Dataset):
    def __init__(self, data_folder, padsize=50):
        self.local_imgflist = main_hyper.load_dir_structs(data_folder)
        self.padsize = padsize
        # config.gpu_options.allow_growth = True

    def __getitem__(self, index):
        if os.path.splitext(self.local_imgflist[index])[1] == '':
            print('Empty file', self.local_imgflist[i])
            return None
        
        ori_img, alpha = main_hyper.read_png4(self.local_imgflist[index], input_size=shape, img_mean=main_hyper.IMG_MEAN)
        pad_img = tf.pad(ori_img, [[self.padsize, self.padsize], [self.padsize, self.padsize], [0, 0]], mode='REFLECT')
        pad_img_ = sess.run(pad_img)
        cur_embed = model.test(pad_img_)
        cur_embed = np.squeeze(cur_embed)
        cur_embed = cur_embed[self.padsize: (cur_embed.shape[0]-self.padsize), self.padsize: (cur_embed.shape[1]-self.padsize), :]
        ori_img = ori_img.eval(session=sess)
        alpha = alpha.eval(session=sess)
        ori_img = ori_img / 255.
        alpha = alpha / 255.
        #assert ori_img.shape[0] == cur_embed.shape[0] and ori_img.shape[1] == cur_embed.shape[1]
        return np2Tensor(cur_embed), np2Tensor(ori_img), np2Tensor(alpha)

    def __len__(self):
        return len(self.local_imgflist)
    
    