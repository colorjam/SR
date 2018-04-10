from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def img2tensor(l, test, crop_size=256):
    def _img2tensor(i, img):
        size = int(crop_size) * 2**i 
        if test:
            return ToTensor()(img)
        else:
            return Compose([
                CenterCrop(size),
                ToTensor(),
            ])(img)

    return (_img2tensor(i, _l) for i, _l in enumerate(l))