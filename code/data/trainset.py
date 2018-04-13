from os import listdir
from os.path import join
from PIL import Image

from data.common import is_image_file, img_transform, set_channel

from torch.utils.data import Dataset

class Trainset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dir_hr = join(args.dir_datasets, args.data_train + '/HR')
        self.images_hr = [join(self.dir_hr, x) for x in listdir(self.dir_hr) if is_image_file(x)]
        
    def __getitem__(self, idx):
        hr = Image.open(self.images_hr[idx])
        hr = set_channel([hr], self.args.n_channel)[0]
        input = img_transform(hr, self.args.crop_size, 4, is_normalize=True)
        target_x2 = img_transform(hr, self.args.crop_size, 2)
        target_x4 = img_transform(hr, self.args.crop_size, 1)

        return input, target_x2, target_x4

    def __len__(self):
        return len(self.images_hr)



