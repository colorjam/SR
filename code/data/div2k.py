from os import scandir
from os.path import join
from PIL import Image

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
        input = _transform(Image.open(self.images_lr[-1][idx]), self.args.crop_size)

        # target: x2 | x4 | x2 + x4
        target = []
        if len(upscale) > 1: # multiple scale
            target.append(_transform(Image.open(self.images_lr[0][idx]), self.args.crop_size, upscale[0]))
        hr = _transform(Image.open(self.images_hr[idx]), self.args.crop_size, upscale[-1])
        target.append(hr)

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
