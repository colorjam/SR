from PIL import Image
import argparse
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import numpy as np

dir_dataset = '../datasets/Set5/HR'
images = [join(dir_dataset, x) for x in listdir(dir_dataset)]
imglist = []




img = Image.open(images[0])
img = transforms.ToTensor()(img).numpy().transpose((1, 2, 0))

def transform(img, op, axes=(0 ,1)):
    ret = None
    if op == 'vflip':
        ret = np.fliplr(img)
    if op == 'hflip':
        ret = np.flipud(img)
    if op == 'rotate':
        ret = np.rot90(img, axes=axes)
    return ret

titles = ['original img','rotate90','rotate180', 'rotate270',
'flipped img', 'flipped rotate90', 'flipped rotate180', 'flipped rotate270']

def show(imglist):
    plt.figure()
    for i, img in enumerate(imglist):
        ax = plt.subplot(2, 4, i+1)
        ax.set_title(titles[i])
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

def rotate(inputlist):
    for i in range(3):
        inputlist.extend([np.rot90(inputlist[-1])])

inputlist = [img]
rotate(inputlist)
inputlist.extend([np.flipud(img)])
rotate(inputlist)

outputlist = inputlist

for i in range(len(inputlist)):
    k = i % 4
    if k > 0:
        outputlist[i] = np.rot90(inputlist[i], k=k, axes=(1, 0))
    if i > 3:
        outputlist[i] = np.flipud(inputlist[i])

# print(outputlist)

show(outputlist)


