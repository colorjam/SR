from os import listdir
from os.path import join
from PIL import Image

from data.common import is_image_file, img2tensor

from torch.utils.data import Dataset

class Demo():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dir_demo = join(args.dir_dataset, args.data_test)
        self.images_demo = [join(self.dir_demo, x) for x in listdir(self.dir_demo) if is_image_file(x)]
   
    def __getitem__(self, idx):
        input = Image.open(self.images_demo[idx])
        return img2tensor([input], self.args.test), -1, -1
        
    def __len__(self):
        return len(self.images_demo)

