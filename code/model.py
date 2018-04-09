import torch
import torch.nn as nn
import torch.nn.init as init

import math

def default_conv(in_channelss, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channelss, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, act=nn.ReLU(True), res_scale=1):
        super().__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform(m.weight.data)
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):

        modules = []
        modules.append(conv(n_feat, 4 * n_feat, 3, bias))
        modules.append(nn.PixelShuffle(2))
        if act: modules.append(act())

        super().__init__(*modules)

class Net(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        n_colors = 3
        act = nn.ReLU(True)
        self.upscale = args.upscale
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        # define body module
        modules_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1) \
            for _ in range(n_resblock)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        modules_tail = [
            Upsampler(conv, self.upscale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        SR_2x = self.head(x)
        res = self.body(SR_2x)
        res += SR_2x
        SR_2x = self.tail(res)

        SR_4x = self.head(SR_2x)
        res = self.body(SR_4x)
        res += SR_4x
        SR_4x = self.tail(res)

        return SR_2x, SR_4x

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        n = X.size()[0]
        diff = X - Y
        # print(diff)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error) 
        return loss

    

  


