import random
from os import scandir
from os.path import join
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import cv2

from data.common import is_image_file

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
        idx = idx % self.n_train
        upscale = self.args.upscale

        def _transform(img):
            return ToTensor()(img)
        
        # input: x2 | x4
        input = Image.open(self.images_lr[-1][idx])

        # target: x2 | x4 | x2 + x4
        target = []
        if len(upscale) > 1: # multiple scale
            target.append(Image.open(self.images_lr[0][idx]))
        hr = Image.open(self.images_hr[idx])
        target.append(hr)

        # crop images
        input, target = self._get_crop(input, target)
        
        # data augmentation
        if self.args.aug:
            input, target = self.augment([input, target])

        if self.args.random:
            random_factor = random.random()
            input, target = self.environment_factor([input, target], random_factor)

        # transform
        input = _transform(input)
        if len(upscale) > 1:
            target[0] = _transform(target[0])
        target[-1] = _transform(target[-1])

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

    def _get_crop(self, input, target):
        upscale = self.args.upscale
        crop_size = self.args.crop_size

        def _crop(img, crop_size):
            iw, ih = img.size
            left, right = (iw - crop_size) / 2, (iw + crop_size) / 2
            top, bottom = (ih - crop_size) / 2, (ih + crop_size) / 2
            return img.crop((left, top, right, bottom))

        input = _crop(input, crop_size)
        if len(upscale) > 1:
            target[0] = _crop(target[0], crop_size * upscale[0])
        target[-1] = _crop(target[-1], crop_size * upscale[-1])

        return input, target


    def augment(self, l, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if type(img) == list:
                return [_augment(i) for i in img]
                
            if hflip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if vflip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if rot90: img = img.transpose(Image.ROTATE_90)
            
            return img

        return [_augment(_l) for _l in l]

    def random_color(self, l, random_factor, saturation=True, brightness=True, contrast=True, sharpness=True):
        saturation = saturation and random.random() < 0.5
        brightness = brightness and random.random() < 0.5
        contrast = contrast and random.random() < 0.5
        sharpness = sharpness and random.random() < 0.5
        
        def _random(img):
            if type(img) == list:
                return [_random(i) for i in img]

            if saturation:   
                img = ImageEnhance.Color(img).enhance(random_factor)  
            if brightness:
                img = ImageEnhance.Brightness(img).enhance(random_factor) 
            if contrast: 
                img = ImageEnhance.Contrast(img).enhance(random_factor) 
            if sharpness:
                img = ImageEnhance.Sharpness(img).enhance(random_factor) 

            return img

        return [_random(_l) for _l in l] 

    def environment_factor(self, l, random_factor):
        def _factor(img):
            if type(img) == list:
                return [_factor(i) for i in img]

            im = np.asarray(img)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = hsv[:,:,0] * (0.8 + random_factor * 0.2)
            hsv[:,:,1] = hsv[:,:,1] * (0.3 + random_factor * 0.7)
            hsv[:,:,2] = hsv[:,:,2] * (0.2 + random_factor * 0.8)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return Image.fromarray(np.uint8(im))

        return [_factor(_l) for _l in l] 
