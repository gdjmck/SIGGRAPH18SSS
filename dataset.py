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
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))    
    return tensor

class InputData(data.Dataset):
    def __init__(self, data_folder, padsize=50):
        self.local_imgflist = main_hyper.load_dir_structs(data_folder)
        self.padsize = padsize
        config = tf.ConfigProto(device_count = {'GPU': 0})
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model = HyperColumn_Deeplabv2(self.sess, args)
        self.model.load(args.snapshot_dir)

    def __getitem__(self, index):
        print('__getitem__', index)
        if os.path.splitext(self.local_imgflist[index])[1] == '':
            print('Empty file', self.local_imgflist[i])
            return None
        
        sess = tf.Session()
        print(self.local_imgflist[index])
        ori_img, alpha = main_hyper.read_png4(self.local_imgflist[index], input_size=shape, img_mean=main_hyper.IMG_MEAN)
        print('read_png4', index, ori_img.shape)
        pad_img = tf.pad(ori_img, [[self.padsize, self.padsize], [self.padsize, self.padsize], [0, 0]], mode='REFLECT')
        print('padded')
        pad_img_ = sess.run(pad_img)
        print('sent to extract feat', index, pad_img_.shape)
        cur_embed = self.model.test(pad_img_)
        print('predict ', index)
        cur_embed = np.squeeze(cur_embed)
        cur_embed = cur_embed[self.padsize: (cur_embed.shape[0]-self.padsize), self.padsize: (cur_embed.shape[1]-self.padsize), :]
        ori_img = ori_img.eval(session=sess)
        alpha = alpha.eval(session=sess)
        #assert ori_img.shape[0] == cur_embed.shape[0] and ori_img.shape[1] == cur_embed.shape[1]
        return np2Tensor(cur_embed), np2Tensor(ori_img), np2Tensor(alpha)

    def __len__(self):
        return len(self.local_imgflist)
    
    