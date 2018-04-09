from option import args
import numpy as np
from math import log10

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils   
from data import get_loader
from logger import Logger, to_np

ckp = utils.checkpoint(args)
logger = Logger(args.log_file)

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loading datasets')
loader_train, loader_test = get_loader(args)

print('===> Building model')

model, criterion, optimizer = ckp.get_model()


if cuda:
    model = model.cuda()
    criterion = criterion.cuda()


def train(epoch):

    def _train_forward(x):
        if args.aug:
            return utils.train_transform(x, model)
        else:
            return model(x)

    epoch_loss = 0
    timer_data, timer_model = utils.timer(), utils.timer()
    for iteration, batch in enumerate(loader_train, 1):
        input, target_2x, target_4x = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda()
            target_2x = target_2x.cuda()
            target_4x = target_4x.cuda()

        timer_data.hold()
        timer_model.tic()
        
        optimizer.zero_grad()


        prediction_2x, prediction_4x = _train_forward(input)
        
        # print('prediction_2x', prediction_2x.data[0].shape)
        # print('target_2x', target_2x.data[0].shape)

        loss_2x = criterion(prediction_2x, target_2x)
        loss_4x = criterion(prediction_4x, target_4x)
        loss = loss_2x + loss_4x
        epoch_loss += loss.data[0]
        loss.backward()

        optimizer.step()

        psnr_2x = utils.calc_PSNR(target_2x.data[0], prediction_2x.data[0], 2)
        psnr_4x = utils.calc_PSNR(target_4x.data[0], prediction_4x.data[0], 4)

        timer_model.hold()


        ckp.write_log('===> Epoch[{}]({}/{}): loss: {:.4f} psnr_2x: {:.4f} dB pnsr_4x: {:.4f} dB \
            {:.1f}+{:.1f}s'.format(
            epoch, iteration, len(loader_train), loss.data[0], 
            psnr_2x, psnr_4x, timer_model.release(), timer_data.release()))

        timer_data.tic()

        #============ TensorBoard logging ============#
        iter = iteration + (epoch - 1) * len(loader_train)

        # Log the scalar values
        info = {
            'loss': loss.data[0],
            'psnr_2x': psnr_2x.data[0],
            'psnr_4x': psnr_4x.data[0]
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, iter)

        # Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), iter)
            logger.histo_summary(tag + '/grad', to_np(value.grad), iter)

    ckp.write_log('===> Epoch {} Complete: Avg. loss: {:.4f}'.format(
        epoch, epoch_loss / len(loader_train)))

def test():
    ckp.write_log('===> Evaluation')
    avg_psnr_2x = 0.0
    avg_psnr_4x = 0.0
    timer_test = utils.timer()
    for iteration, batch in enumerate(loader_test, 1):
        # print(iteration)
        input, target_2x, target_4x = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda()
            target_2x = target_2x.cuda()
            target_4x = target_4x.cuda()

        save_list = [input, target_2x, target_4x]

        prediction_2x, prediction_4x = model(input)

        save_list.extend([prediction_2x, prediction_4x])
        psnr_2x = utils.calc_PSNR(target_2x.data[0], prediction_2x.data[0], 2)
        psnr_4x = utils.calc_PSNR(target_4x.data[0], prediction_4x.data[0], 4)
        if args.test:
            ckp.write_log('===> Image{} PSNR_2x: {:.4f} dB'.format(iteration, psnr_2x))
            ckp.write_log('===> Image{} PSNR_4x: {:.4f} dB'.format(iteration, psnr_4x))
            ckp.save_result(iteration, save_list)

    ckp.write_log("===> Total time: {:.1f}s".format(timer_test.toc()))


if args.test:
    test() # evaluation model
else:
    start = 1 if args.load == -1 else args.load + 1
    end = args.epochs + start
    for epoch in range(start, end):
        train(epoch)
        test()
        ckp.save_model(model, epoch)

ckp.done()
