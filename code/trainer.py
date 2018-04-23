import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils   
from tqdm import tqdm
from logger import Logger, to_np

class Trainer():
    def __init__(self, args, loader, model, criterion, optimizer, ckp):
        self.args = args

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ckp = ckp

        self.best_psnr = 0.0
        self.best_epoch = 1

    def train(self, epoch):
        def _train_forward(x):
            if self.args.aug:
                return utils.augmentation(x, self.model, self.args.upscale)
            else:
                return self.model(x)

        logger = Logger(self.args.log_file)
        epoch_loss = 0.0
        timer_data, timer_model = utils.timer(), utils.timer()
        for iteration, (input, hr) in enumerate(self.loader_train, 1):
            input, hr = self.prepare([input, hr])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(input)

            loss = 0.0
            for i in range(len(hr)):
                loss += self.criterion(sr[i], hr[i])
            
            epoch_loss += loss.data[0]
            loss.backward()

            self.optimizer.step()

            timer_model.hold()

            if iteration % self.args.print_freq == 0:
                self.ckp.write_log(
                    '=> Epoch[{}]({}/{}):\t'
                    'Loss: {:.4f}\t'
                    'Time: {:.1f}+{:.1f}s'.format(
                    epoch, iteration, len(self.loader_train), loss.data[0], 
                    timer_model.release(), timer_data.release()))

            timer_data.tic()
        
        self.ckp.write_log('=> Epoch {} Complete: Avg. loss: {:.4f}'.format(
            epoch, epoch_loss / len(self.loader_train)))
        
    def test(self, epoch=10):
        self.ckp.write_log('=> Evaluation...')
        timer_test = utils.timer()
        upscale = self.args.upscale
        avg_psnr = {}
        avg_ssim = {}

        for scale in upscale:
            avg_psnr[scale] = 0.0
            avg_ssim[scale] = 0.0
        
        for iteration, (input, hr) in enumerate(self.loader_test, 1):

            has_target = type(hr) == list # if test on demo

            if has_target:
                input, hr = self.prepare([input, hr])
            else:
                input = self.prepare([input])[0]
           
            sr = self.model(input)

            save_list = [*sr, input]

            if has_target:
                save_list.extend(hr)

                psnr = {}
                ssim = {}
                for i, scale in enumerate(upscale):
                    psnr[scale] = utils.calc_psnr(hr[i], sr[i], int(scale))
                    ssim[scale] = utils.calc_ssim(hr[i], sr[i]) 
                    avg_psnr[scale] += psnr[scale]
                    avg_ssim[scale] += ssim[scale]

            if self.args.save:
                if has_target:
                    for i, scale in enumerate(upscale):
                        self.ckp.write_log('=> Image{} PSNR_x{}: {:.4f}'.format(iteration, scale, psnr[scale]))
                        self.ckp.write_log('=> Image{} SSIM_x{}: {:.4f}'.format(iteration, scale, ssim[scale]))
                self.ckp.save_result(iteration, save_list)

        if has_target:
            for scale, value in avg_psnr.items():
                self.ckp.write_log("=> PSNR_x{}: {:.4f}".format(scale, value/len(self.loader_test)))
                self.ckp.write_log("=> SSIM_x{}: {:.4f}".format(scale, avg_ssim[scale]/len(self.loader_test)))
                
        self.ckp.write_log("=> Total time: {:.1f}s".format(timer_test.toc()))

        if not self.args.test:
            self.ckp.save_model(self.model, 'latest')
            cur_psnr = avg_psnr[upscale[-1]]
            if self.best_psnr < cur_psnr:
                self.best_psnr = cur_psnr
                self.best_epoch = epoch
                self.ckp.save_model(self.model, '{}_best'.format(self.best_epoch))

    def prepare(self, l):
        def _prepare(tensor):       

            if type(tensor) == list:
                return [_prepare(t) for t in tensor]

            if self.args.cuda:
                tensor = tensor.cuda()
            return Variable(tensor)

        return [_prepare(_l) for _l in l]

  

        


        

        

        

        



