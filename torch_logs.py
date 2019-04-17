from tensorboardX import SummaryWriter
import os
import torch
import datetime

def timestamp():
    return str(datetime.datetime.now()).replace(' ', '_')

class TrainLog():
    def __init__(self, args):
        self.args = args
        
        self.step = 0
        self.save_dir = args.saveDir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
            
        summary_dir = os.path.join(self.save_dir, timestamp())
        os.makedirs(summary_dir)
        self.summary = SummaryWriter(summary_dir)
            
    def next_step(self):
        self.step += 1
        
    def add_scalar(self, scalar_name, scalar, step=None):
        if step is None:
            step = self.step
        self.summary.add_scalar(scalar_name, scalar, step)
        
    def add_histogram(self, var_name, value, step=None):
        if step is None:
            step = self.step
        self.summary.add_histogram(var_name, value, step)
            
    def load_model(self, model):
        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_model_dir)
        if self.args.without_gpu: # 用cpu载入模型到内存
            ckpt = torch.load(lastest_out_path, map_location='cpu')
        else: # 模型载入到显存
            ckpt = torch.load(lastest_out_path)
        state_dict = ckpt['state_dict'].copy()
        for key in ckpt['state_dict']:
            if key not in model.state_dict():
                print('missing key:\t', key)
                state_dict.pop(key)
        ckpt['state_dict'] = state_dict
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'], strict=False)
        self.step = ckpt['step']
        #self.step_cnt = 1
        print("=> loaded checkpoint '{}' (epoch {}  total step {})".format(lastest_out_path, ckpt['epoch'], self.step))

        return start_epoch, model
    
    def save_model(self, model, epoch):
        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_model_dir)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'step': self.step
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_model_dir)
        torch.save(
            model,
            model_out_path)
        