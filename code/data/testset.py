from os import listdir
from os.path import join
from PIL import Image

from data.common import is_image_file, img_transform

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize

class Testset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dir_hr = join(args.dir_datasets, args.data_test + '/HR')
        self.dir_lr_x2 = join(args.dir_datasets, args.data_test + '/LR/X2')
        self.dir_lr_x4 = join(args.dir_datasets, args.data_test + '/LR/X4')
        
        self.images_hr = [join(self.dir_hr, x) for x in listdir(self.dir_hr) if is_image_file(x)]
        self.images_lr_x2 = [join(self.dir_lr_x2, x) for x in listdir(self.dir_lr_x2) if is_image_file(x)]
        self.images_lr_x4 = [join(self.dir_lr_x4, x) for x in listdir(self.dir_lr_x4) if is_image_file(x)]
        
    def __getitem__(self, idx):
        def _transform(img, crop_size):
            return Compose([
                CenterCrop(crop_size),
                ToTensor()
            ])(img)
        def _normalize(img):
            return Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
            ])(img)
        lr_x4 = Image.open(self.images_lr_x4[idx])
        lr_x2 = Image.open(self.images_lr_x2[idx])
        hr = Image.open(self.images_hr[idx])
        w, h = lr_x2.size
        input, target_x2, target_x4 = ToTensor()(lr_x4), ToTensor()(lr_x2), _transform(hr, (2*h, 2*w))
        return input, target_x2, target_x4

    def __len__(self):
        return len(self.images_hr)



