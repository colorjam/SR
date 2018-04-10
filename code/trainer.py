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

    def train(self, epoch):
        def _train_forward(x):
            if self.args.aug:
                return utils.train_transform(x, self.model)
            else:
                return self.model(x)

        logger = Logger(self.args.log_file)
        epoch_loss = 0.0
        timer_data, timer_model = utils.timer(), utils.timer()
        for iteration, (input, hr_x2, hr_x4) in enumerate(self.loader_train, 1):
            input, hr_x2, hr_x4 = self.prepare([input, hr_x2, hr_x4])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr_x2, sr_x4 = _train_forward(input)

            loss_2x = self.criterion(sr_x2, hr_x2)
            loss_4x = self.criterion(sr_x4, hr_x4)
            loss = loss_2x + loss_4x
            epoch_loss += loss.data[0]
            loss.backward()

            self.optimizer.step()

            timer_model.hold()

            self.ckp.write_log('=> Epoch[{}]({}/{}): loss: {:.4f} {:.1f}+{:.1f}s'.format(
                epoch, iteration, len(self.loader_train), loss.data[0], 
                timer_model.release(), timer_data.release()))

            timer_data.tic()

            #============ TensorBoard logging ============#

            iter = iteration + (epoch - 1) * len(self.loader_train)

            # Log the scalar values
            info = {
                'loss': loss.data[0]
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter)

            # Log values and gradients of the parameters (histogram)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), iter)
                logger.histo_summary(tag + '/grad', to_np(value.grad), iter)

        self.ckp.write_log('=> Epoch {} Complete: Avg. loss: {:.4f}'.format(
            epoch, epoch_loss / len(self.loader_train)))
        
    def test(self, epoch=10):
        self.ckp.write_log('=> Evaluation')
        timer_test = utils.timer()
        avg_psnr_2x = 0.0
        avg_psnr_4x = 0.0

        for iteration, (input, hr_x2, hr_x4) in enumerate(self.loader_test, 1):

            has_target = isinstance(hr_x2, object)
            if has_target:
                input, hr_x2, hr_x4 = self.prepare([input, hr_x2, hr_x4])
            else:
                input = self.prepare([input])[0]

            sr_x2, sr_x4 = self.model(input)

            save_list = [sr_x2, sr_x4, input]
            if has_target:
                save_list.extend([hr_x2, hr_x4])

            psnr_2x = utils.calc_PSNR(hr_x2.data[0], sr_x2.data[0], 2)
            psnr_4x = utils.calc_PSNR(hr_x4.data[0], sr_x4.data[0], 4)
            avg_psnr_2x += psnr_2x
            avg_psnr_4x += psnr_4x

            if self.args.save:
                self.ckp.write_log('=> Image{} PSNR_2x: {:.4f}'.format(iteration, psnr_2x))
                self.ckp.write_log('=> Image{} PSNR_4x: {:.4f}'.format(iteration, psnr_4x))
                self.ckp.save_result(iteration, save_list)

        self.ckp.write_log("=> PSNR_2x: {:.4f} PSNR_4x: {:.4f} Total time: {:.1f}s".format(
            avg_psnr_2x / len(self.loader_test), avg_psnr_4x / len(self.loader_test), timer_test.toc()))

        if not self.args.test:
            self.ckp.save_model(self.model, epoch)

    def prepare(self, l):
        def _prepare(tensor):
            if self.args.cuda:
                tensor = tensor.cuda()
            return Variable(tensor)

        return (_prepare(_l) for _l in l)

  

        


        

        

        

        



