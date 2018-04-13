import numpy as np
import skimage.color as sc
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def img_transform(img, crop_size, upscale=1,is_normalize=False):
    if is_normalize:
        return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale),
        ToTensor(),
        normalize
    ])(img)
    else:
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // upscale),
            ToTensor(),
        ])(img)

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]