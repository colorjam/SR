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
        _make_dir(self.dir + '/results_{}'.format(self.args.loss))

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
        filename = '{}/results_{}/{}_'.format(self.dir, self.args.loss, idx)
        scale = self.args.upscale
        if len(scale)>1:
            postfix = ('SR_x2', 'SR_x4', 'LR', 'HR_x2', 'HR_x4')
        else:
            postfix = ('SR_x{}'.format(scale[0]), 'LR', 'HR_x{}'.format(scale[0]))

        for v, n in zip(save_list, postfix):
            tu.save_image(v.data[0], '{}{}.png'.format(filename, n))

    def done(self):
        self.log_file.write('\n')
        self.log_file.close()

def quantize(img):
    return img.clamp(0, 255).round()

def calc_psnr(sr, hr, scale, benchmark=False):
    '''
        we measure PSNR on luminance channel only
    '''
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
    sr = sr.data[0].squeeze().numpy().transpose(1, 2, 0)
    hr = hr.data[0].squeeze().numpy().transpose(1, 2, 0)
    return ssim(sr, hr, multichannel=True)
    
def augmentation(input, model, upscale):

    def _totensor(x):
        return Variable(torch.Tensor(x.copy()))

    def _rotate(input, k=1, axes=(2, 3)): # (n, c, h, w)
        img = input.data.cpu().numpy()
        return _totensor(np.rot90(img, k=k, axes=axes).copy())

    def _flip(input, axis=2):
        img = input.data.cpu().numpy()
        return _totensor(np.flip(img, axis=axis).copy())

    inputlist = [input]
    # rotate
    for i in range(3):
        inputlist.extend([_rotate(inputlist[-1], axes=(2, 3))])
    # flip
    inputlist.extend([_flip(input)])
    for i in range(3):
        inputlist.extend([_rotate(inputlist[-1], axes=(2, 3))])

    outputlist = [[] for _ in upscale]
    for input in inputlist:
        output = model(input)
        for i, o in enumerate(output):
            outputlist[i].append(o)

    for i in range(len(outputlist[0])):
        k = i % 4
        if k > 0:
            for j in range(len(upscale)):
                outputlist[j][i] = _rotate(outputlist[j][i], k=k, axes=(3, 2))
        if i > 3:
            for j in range(len(upscale)):
                outputlist[j][i] = _flip(outputlist[j][i])

    output = [[] for _ in upscale]
    for j in range(len(upscale)):
        output[j] = reduce((lambda x, y: x + y), outputlist[j]) / len(outputlist[j])

    return output

            




    






