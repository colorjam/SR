import torch
import torch.nn as nn
import math

class Upsampler(nn.Sequential):
    def __init__(self, scale, in_channels, bias=True):
        modules = []
        modules.append(nn.Conv2d(in_channels, scale**2*in_channels, kernel_size=3, padding=1, bias=True))
        modules.append(nn.PixelShuffle(scale))

        super().__init__(*modules)

def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2), # kernel, stride, padding
        4: (8, 4, 2),
        8: (12, 8, 2)
    }[scale]
    if up:
        conv_f = Upsampler(
            scale, in_channels
        )
    else:
        conv_f = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride, padding
    )

    return conv_f

class BackProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super().__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.ReLU()
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.ReLU()
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.ReLU()
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.ReLU()
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        #print('a_0', a_0.shape)
        b_0 = self.conv_2(a_0)
        #print('b_0', b_0.shape)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)
        #print('a_1', a_1.shape)
        out = a_0.add(a_1)

        return out

class DBPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scale = args.upscale[0]
        self.args = args

        n0 = args.n_init # number of filters used in the initial LR features extraction
        self.nr = args.n_feats # number of filters used in each projection unit.
        self.depth = args.n_blocks

        # 1 Initial
        initial = [
            nn.Conv2d(args.n_colors, n0, 3, padding=1), # constuct initial feature-maps
            nn.ReLU(),
            nn.Conv2d(n0, self.nr, 1), # reduce the dimention
            nn.ReLU()
        ]
        self.initial = nn.Sequential(*initial)

        # 2 Back-projection stages
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        for i in range(self.depth):
            self.upmodules.append(
                BackProjection(self.nr, self.nr, self.scale, True, i > 1)
            )
        for i in range(self.depth - 1):
            self.downmodules.append(
                BackProjection(self.nr, self.nr, self.scale, False, i > 1)
            )

        # 3 Reconstruction
        reconstruction = [
            # nn.Conv2d(self.depth * self.nr, args.n_colors, 3, padding=1) 
            nn.Conv2d(self.nr, args.n_colors, 3, padding=1) 
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        if args.init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.ConvTranspose2d):
        	        torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        l = self.initial(x)
        h_list = []

        n, f, h, w = l.shape
        h = torch.zeros(n, f, h * self.scale, w * self.scale)

        for i in range(self.depth - 1):
            h += self.upmodules[i](l)
            # h_list.append(self.upmodules[i](l))
            # l = self.downmodules[i](h_list[-1])
            l = self.downmodules[i](h)
        
        # h_list.append(self.upmodules[-1](l))
        # out = self.reconstruction(torch.cat(h_list, dim=1))
        h = self.upmodules[-1](l)
        out = self.reconstruction(h)

        return [out]

