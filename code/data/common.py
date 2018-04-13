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
