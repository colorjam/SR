import random
from os import scandir
from os.path import join
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np

from data.common import is_image_file, set_channel, train_transform, test_transform

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize

class DIV2K(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.args = args
        self.train = train
        self.dir_hr = join(args.dir_datasets + '/DIV2K/HR')
        self.dir_lr = [join(args.dir_datasets + '/DIV2K/LR/X' + str(scale)) for scale in args.upscale]
        
        self.n_train = args.n_train
        self.n_test = 20

        if train:
            self.images_hr = [entry.path for entry in scandir(self.dir_hr) if is_image_file(entry.name)][:self.n_train]
        else:
            self.images_hr = [entry.path for entry in scandir(self.dir_hr) if is_image_file(entry.name)][self.n_train:self.n_train + self.n_test]
        
        self.images_lr = self._get_lr()

    def __getitem__(self, idx):
        upscale = self.args.upscale

        if self.train:
            _transform = train_transform
        else:
            _transform = test_transform
        
        # input: x2 | x4
        input = Image.open(self.images_lr[-1][idx])

        # target: x2 | x4 | x2 + x4
        target = []
        if len(upscale) > 1: # multiple scale
            target.append(Image.open(self.images_lr[0][idx]))
        hr = Image.open(self.images_hr[idx])
        target.append(hr)

        # data augmentation
        if self.args.aug:
            input, target = self.augment([input, target])

        if self.args.random:
            input, target = self.random_color([input, target])

        # transform
        input = _transform(input, self.args.crop_size)
        if len(upscale) > 1:
            target[0] = _transform(target[0], self.args.crop_size, upscale[0])
        target[-1] = _transform(target[-1], self.args.crop_size, upscale[-1])

        return input, target

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test

    def _get_lr(self):
        list_lr = [[] for _ in self.args.upscale]
        for i, scale in enumerate(self.args.upscale):
            for filename in self.images_hr:
                filename = filename.split('/')[-1].split('.')[0]
                list_lr[i].append(join(self.dir_lr[i], '{}x{}.png'.format(filename, str(scale))))
        return list_lr

    def augment(self, l, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if type(img) == list:
                return [_augment(i) for i in img]
                
            # print("img.shape", img.shape)
            if hflip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if vflip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if rot90: img = img.transpose(Image.ROTATE_90)
            
            return img

        return [_augment(_l) for _l in l]

    def random_color(self, l, saturation=True, brightness=True, contrast=True, sharpness=True):
        saturation = saturation and random.random() < 0.5
        brightness = brightness and random.random() < 0.5
        contrast = contrast and random.random() < 0.5
        sharpness = sharpness and random.random() < 0.5
        
        def _random(img):
            if type(img) == list:
                return [_random(i) for i in img]

            if saturation:   
                random_factor = random.uniform(1, 3)  
                img = ImageEnhance.Color(img).enhance(random_factor)  
            if brightness:
                random_factor = random.uniform(1, 2) 
                img = ImageEnhance.Brightness(img).enhance(random_factor) 
            if contrast: 
                random_factor = random.uniform(1, 2) 
                img = ImageEnhance.Contrast(img).enhance(random_factor) 
            if sharpness:
                random_factor = random.uniform(1, 3)  
                img = ImageEnhance.Sharpness(img).enhance(random_factor) 

            return img

        return [_random(_l) for _l in l] 
