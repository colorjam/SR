import os
import datetime
import time
from functools import reduce

import math
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tu
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose

from model import Net, L1_Charbonnier_loss, Squeeze_loss

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
        _make_dir(self.dir + '/results')

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
        model = Net(self.args)
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
        
        # criterion = nn.L1Loss()
        loss = self.args.loss
        if loss == 'MSE':
            criterion = nn.MSELoss()
        elif loss == 'L1':
            criterion = nn.L1Loss()
        elif loss == 'Charbonnier':
            criterion = L1_Charbonnier_loss()
        elif loss == 'Perceptual':
            criterion = Squeeze_loss()

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        return model, criterion, optimizer

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')
    
    def save_model(self, model, epoch):
        torch.save(model.state_dict(), '{}/model/model_{}.pt'.format(self.dir, epoch))
    
    def save_result(self, idx, save_list):
        filename = '{}/results/{}_'.format(self.dir, idx)
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

def rgb2ycbcr(rgb): # Tensor of [N, C, H, W]
    def _rgb2cbcr(rgb):
        rgb = ToPILImage()(rgb.squeeze())
        y, cb, cr = rgb.convert('YCbCr').split()
        y = Variable(ToTensor()(y).unsqueeze(0))
        return y, cb, cr
    
    return [_rgb2cbcr(img) for img in rgb]

def ycbcr2rgb(y, cb, cr):
    y = y.squeeze().numpy()
    cb, cr = Resize(y.shape)(cb), Resize(y.shape)(cr)
    y = (y * 255.0).clip(0, 255)
    y = Image.fromarray(np.uint8(y), mode='L')
    rgb = Image.merge('YCbCr', [y, cb, cr]).convert('RGB')
    return Variable(ToTensor()(rgb).unsqueeze(0).contiguous())

# def calc_PSNR(input, target, scale):
#     # evaluate these datasets in y channel only
#     c, h, w = input.size()
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
    '''
        Here we assume quantized(0-255) arguments.
        For Set5, Set14, B100, Urban100 dataset,
        we measure PSNR on luminance channel only
    '''
    diff = (sr - hr).data.div(255)
    # if benchmark:
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
    
def train_transform(input, model):

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

    outputlist = [model(aug) for aug in inputlist]
    outputlist_2x, outputlist_4x = [list(a) for a in zip(*outputlist)]

    for i in range(len(outputlist_2x)):
        k = i % 4
        if k > 0:
            outputlist_2x[i] = _rotate(outputlist_2x[i], k=k, axes=(3, 2))
            outputlist_4x[i] = _rotate(outputlist_4x[i], k=k, axes=(3, 2))
        if i > 3:
            outputlist_2x[i] = _flip(outputlist_2x[i])
            outputlist_4x[i] = _flip(outputlist_4x[i])

    output_2x = reduce((lambda x, y: x + y), outputlist_2x) / len(outputlist_2x)
    output_4x = reduce((lambda x, y: x + y), outputlist_4x) / len(outputlist_4x)

    return output_2x, output_4x

            




    






