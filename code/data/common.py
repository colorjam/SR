import numpy as np
import skimage.color as sc
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def img_transform(img, crop_size, upscale=1):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale),
        ToTensor(),
    ])(img)

def train_transform(img):
    return ToTensor()(img)

def test_transform(img, crop_size, upscale=1):
    return ToTensor()(img)

def set_channel(l, n_channel):
    def _set_channel(img):
        if np.array(img).ndim == 2:
            img = np.expand_dims(img, axis=2)
        print(img.size)
        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


