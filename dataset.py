import tensorflow as tf
from parse_opt import get_arguments
from deeplab_resnet import HyperColumn_Deeplabv2, read_data_list
import os
import torch
import torch.utils.data as data
import main_hyper

shape = (250, 250)
args = get_arguments()
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))    
    return tensor

class FeatureData(data.Dataset):
    def __init__(self, data_folder, padsize=50):
        self.local_imgflist = main_hyper.load_dir_structs(data_folder)
        self.padsize = padsize
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model = HyperColumn_Deeplabv2(self.sess, args)
        self.model.load(args.snapshot_dir)

    def __getitem__(self, index):
        if os.path.splitext(self.local_imgflist[i])[1] == '':
			continue
        
        _, ori_img = main_hyper.read_img(self.local_imgflist[index], input_size=shape, img_mean=main_hyper.IMG_MEAN)
        pad_img = tf.pad(ori_img, [[self.padsize, self.padsize], [self.padsize, self.padsize], [0, 0]], mode='REFLECT')
        cur_embed = self.model.test(pad_img.eval())
        cur_embed = np.squeeze(cur_embed)
        print(cur_embed.shape)
        return np2Tensor(cur_embed)

    def __len__(self):
        return len(self.local_imgflist)