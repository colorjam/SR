from os import listdir
from os.path import join
from PIL import Image

from data.common import is_image_file, img2tensor

from torch.utils.data import Dataset

class Benchmark(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dir_hr = join(args.dir_datasets, args.data_train + '/HR')
        self.dir_lr_x2 = join(args.dir_datasets, args.data_test + '/LR/X2')
        self.dir_lr_x4 = join(args.dir_datasets, args.data_test + '/LR/X4')

        self.images_hr = [join(self.dir_hr, x) for x in listdir(self.dir_hr) if is_image_file(x)]
        self.images_lr_x2 = [join(self.dir_lr_x2, x) for x in listdir(self.dir_lr_x2) if is_image_file(x)]
        self.images_lr_x4 = [join(self.dir_lr_x4, x) for x in listdir(self.dir_lr_x4) if is_image_file(x)]

    def __getitem__(self, idx):
        input = Image.open(self.images_lr_x4[idx])
        target_2x = Image.open(self.images_lr_x2[idx])
        target_4x = Image.open(self.images_hr[idx])
        input, target_2x, target_4x = img2tensor([input, target_2x, target_4x], self.args.test, self.args.crop_size)
        return input, target_2x, target_4x

    def __len__(self):
        return len(self.images_hr)



