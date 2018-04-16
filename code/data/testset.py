from os import scandir
from os.path import join, splitext
from PIL import Image

from data.common import is_image_file, set_channel

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize

class Testset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.upscale = args.upscale

        self.dir_hr = join(args.dir_datasets, args.data_test + '/HR')
        self.dir_lr = [join(args.dir_datasets, args.data_test + '/LR/X' + str(scale)) for scale in self.upscale]
        self.images_hr = [entry.path for entry in scandir(self.dir_hr) if is_image_file(entry.name)]
        self.images_lr = self._get_lr()

    def __getitem__(self, idx):
        def _transform(img, crop_size):
            return Compose([
                CenterCrop(crop_size),
                ToTensor()
            ])(img)
        
        # input: x2 | x4
        input_scale = int(self.upscale[-1])
        input = ToTensor()(Image.open(self.images_lr[-1][idx]))
        _, h, w = input.shape

        # target: x2 | x4 | x2 + x4
        target = []
        if len(self.upscale) > 1:
            target.append(ToTensor()(Image.open(self.images_lr[0][idx])))
        hr = _transform(Image.open(self.images_hr[idx]), (input_scale*h, input_scale*w))
        target.append(hr)
        
        return input, target

    def __len__(self):
        return len(self.images_hr)

    def _get_lr(self):
        lr = []
        for i, scale in enumerate(self.upscale):
            list_lr = []
            for filename in self.images_hr:
                filename = filename.split('/')[-1].split('.')[0]
                list_lr.append(join(self.dir_lr[i], '{}x{}.png'.format(filename, str(scale))))
            lr.append(list_lr)
        return lr



