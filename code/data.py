import random
from skimage import io
from PIL import Image

from os import listdir
from os.path import join

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

class DatasetFromFolder(Dataset):
    def __init__(self, args):
        super(DatasetFromFolder, self).__init__()
        self.args = args
        self.dir_dataset = '../datasets/'

        if not args.test:
            self.dir_train = join(self.dir_dataset, args.data_train + '/HR')
            self.images_train = [join(self.dir_train, x) for x in listdir(self.dir_train) if is_image_file(x)]

            self.input_transform = input_transform(args.patch_size, 4)
            self.target_transform_2x = input_transform(args.patch_size, 2)
            self.target_transform_4x = target_transform(args.patch_size)

        else:
            self.dir_test = join(self.dir_dataset, args.data_test + '/LR/X4')
            self.dir_target_2x = join(self.dir_dataset, args.data_test + '/LR/X2')
            self.dir_target_4x = join(self.dir_dataset, args.data_test + '/HR')
        
            self.images_test = [join(self.dir_test, x) for x in listdir(self.dir_test) if is_image_file(x)]
            self.images_target_2x = [join(self.dir_target_2x, x) for x in listdir(self.dir_target_2x) if is_image_file(x)]
            self.images_target_4x = [join(self.dir_target_4x, x) for x in listdir(self.dir_target_4x) if is_image_file(x)]
        

    def __getitem__(self, index):
        
        if not self.args.test:
            input = Image.open(self.images_train[index])
            target = input.copy()
            input, target_2x, target_4x = self.input_transform(input), self.target_transform_2x(target), self.target_transform_4x(target)

        else:
            input = Image.open(self.images_test[index])
            target_2x = Image.open(self.images_target_2x[index])
            target_4x = Image.open(self.images_target_4x[index])
            input, target_2x, target_4x = ToTensor()(input), ToTensor()(target_2x), ToTensor()(target_4x) 

        return input, target_2x, target_4x

    def __len__(self):
        if not self.args.test:
            return len(self.images_train) 
        else:
            return len(self.images_test)

def get_loader(args):
    loader_train = None
    if not args.test:
        train_set = DatasetFromFolder(args)
        loader_train = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.train_batch, shuffle=True)

    test_set = DatasetFromFolder(args)
    loader_test = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)

    return loader_train, loader_test

