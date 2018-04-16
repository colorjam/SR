from os import listdir, scandir
from os.path import join
from PIL import Image
from data.common import is_image_file

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class Demo():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dir_demo = join(args.dir_datasets, args.data_test)
        self.images_demo = [entry.path for entry in scandir(self.dir_demo) if is_image_file(entry.name)]

    def __getitem__(self, idx):
        input = Image.open(self.images_demo[idx])
        return ToTensor()(input), -1
        
    def __len__(self):
        return len(self.images_demo)

    
        

