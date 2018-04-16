from os import scandir
from os.path import join
from PIL import Image

from data.common import is_image_file, train_transform, set_channel

from torch.utils.data import Dataset

class Trainset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dir_hr = join(args.dir_datasets, args.data_train + '/HR')
        self.images_hr = [entry.path for entry in scandir(self.dir_hr) if is_image_file(entry.name)]

    def __getitem__(self, idx):
        upscale = self.args.upscale

        # input: x2 | x4
        hr = Image.open(self.images_hr[idx])
        input = train_transform(hr, self.args.crop_size, upscale[-1])

        # target: x2 | x4 | x2 + x4
        target = []
        if len(upscale) > 1: # multiple upscale
            target.append(train_transform(hr, self.args.crop_size, 2))
        target.append(train_transform(hr, self.args.crop_size, 1))

        return input, target

    def __len__(self):
        return len(self.images_hr)



