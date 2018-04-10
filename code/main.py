from option import args
import numpy as np
from math import log10

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils   
from data import Data
from trainer import Trainer

ckp = utils.checkpoint(args)

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

print('=> Loading datasets...')
loader = Data(args)

model, criterion, optimizer = ckp.get_model()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

t = Trainer(args, loader, model, criterion, optimizer, ckp)

if not args.test:
    start = 1 if args.resume == -1 else args.resume + 1
    end = args.epochs + start
    for epoch in range(start, end):
        t.train(epoch)
        t.test(epoch)      
else:
    t.test()

ckp.done()
