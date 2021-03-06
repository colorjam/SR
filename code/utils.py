import os
import datetime
import time
from functools import reduce

import math
import numpy as np
from PIL import Image
import skimage.color as sc
from skimage.measure import compare_ssim as ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tu
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose

from model import Model

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.dir = '../experiment/' + str(today)

        if args.reset:
            os.system('rm -rf ' + self.dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results_{}'.format(self.args.description))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        self.log_file.write(now + '\n\n')
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def get_model(self):
        """Create model and define loss function (criterion) and optimizer"""
        model = Model(self.args).model
        pre_train = self.args.pre_train

        if pre_train != '.':
            print('=> Loading model from {}'.format(pre_train))
            model.load_state_dict(torch.load(pre_train, map_location=lambda storage, loc: storage))
        else:
            print('=> Building model...')

        resume = self.args.resume
        if resume > 0:
            model.load_state_dict(
                torch.load('{}/model/model_{}.pt'.format(self.dir, resume)))
            print('=> Continue from epoch {}...'.format(resume))
        
        criterion = Model(self.args).criterion

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        return model, criterion, optimizer

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')
    
    def save_model(self, model, epoch):
        torch.save(model.state_dict(), '{}/model/model_{}.pt'.format(self.dir, epoch))
    
    def save_result(self, idx, save_list):
        if self.args.result_path != '.':
            filename = '{}'.format(self.args.result_path)
        else:
            filename = '{}/results_{}/{}_'.format(self.dir, self.args.description, idx)
            
        scale = self.args.upscale
        if len(scale)>1:
            postfix = ('{}_x2'.format(self.args.description), '{}_x4'.format(self.args.description))
        else:
            postfix = ('SR_{}'.format(self.args.description), 'LR', 'HR_{}'.format(self.args.description))

        for v, n in zip(save_list, postfix):
            tu.save_image(v.data[0], '{}{}.png'.format(filename, n))

    def done(self):
        self.log_file.write('\n')
        self.log_file.close()


def rgb2ycbcr(rgb): # Tensor of [N, C, H, W]
    rgb = ToPILImage()(rgb.squeeze())
    y, cb, cr = rgb.convert('YCbCr').split()
    # y = Variable(ToTensor()(y).unsqueeze(0))
    return np.asarray(y)

# def calc_psnr(input, target, scale):
#     # evaluate these datasets in y channel only
#     print(input.data.size())
#     _,c, h, w = input.data.size()
#     diff = input - target
#     shave = scale
#     if c > 1:
#         input_Y = rgb2ycbcr(input.cpu())[0]
#         target_Y = rgb2ycbcr(target.cpu())[0]
#         diff = (input_Y.data - target_Y.data).view(1, h, w)

#     diff = diff[:, shave:(h - shave), shave:(w - shave)]
#     mse = diff.pow(2).mean()
#     psnr = -10 * np.log10(mse)

#     return psnr

def calc_psnr(sr, hr, scale, benchmark=False):
    diff = (sr - hr).data
    # print(diff[:, 0:, :, :])
    shave = scale
    convert = diff.new(1, 3, 1, 1)
    convert[0, 0, 0, 0] = 65.738
    convert[0, 1, 0, 0] = 129.057
    convert[0, 2, 0, 0] = 25.064
    diff.mul_(convert).div_(256)
    diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)

def calc_ssim(sr, hr, benchmark=False):
    y_sr = rgb2ycbcr(sr.data)
    y_hr = rgb2ycbcr(hr.data)
    return ssim(y_hr, y_sr)


            




    






