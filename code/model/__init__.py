import os
from importlib import import_module

import torch.nn as nn

class Model:
    def __init__(self, args):
        module_model = import_module('model.' + args.model.lower())
        # print(module_model)
        self.model = getattr(module_model, args.model)(args)
       
        module_loss = import_module('model.loss')
        loss = args.loss
        if loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss == 'L1':
            self.criterion = nn.L1Loss()
        elif loss == 'Robust':
            self.criterion = module_loss.Robust_loss()
        elif loss == 'Perceptual':
            self.criterion = module_loss.Squeeze_loss()
