# References:
# Moco-v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
import random
import torch
from styleaug import StyleAugmentor


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, augmentor):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        # create style augmentor:
        self.augmentor = augmentor





    def __call__(self, x):
        
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        
        #im1 = transform(im1).unsqueeze(0)
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        #print(im1.shape)
        im1 = im1.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        im2 = im2.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        im1 = self.augmentor(im1)
        im2 = self.augmentor(im2)
        return [im1, im2]


class ThreeCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, base_transform3):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.base_transform3 = base_transform3

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        im3 = self.base_transform3(x)
        return [im1, im2, im3]


class TwoCropsPlusLocalCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, local_transform, local_crop_num):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.local_transform = local_transform
        self.local_crop_num = local_crop_num

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)

        ims_small = []
        for _ in range(self.local_crop_num):
            ims_small.append(self.local_transform(x))

        return [im1, im2, ims_small]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)