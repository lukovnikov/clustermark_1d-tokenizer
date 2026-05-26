import typing

import os

import math

# import wandb

from PIL import Image, ImageFilter

# import pandas as pd

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# from torchvision import transforms
# from torchvision.datasets import CocoDetection, VisionDataset
from torch.utils.data import DataLoader

import numpy as np

import argparse

import uuid

# from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.nn.functional import mse_loss
from PIL import Image

import tempfile

import fire


# for image distortion
distortion_parser = argparse.ArgumentParser(add_help=False)
distortion_parser.add_argument('--r_degree', default=0, type=float)
distortion_parser.add_argument('--jpeg_ratio', default=None, type=int)
distortion_parser.add_argument('--crop_scale_TR', default=None, type=float)
distortion_parser.add_argument('--random_crop_ratio', default=None, type=float)
distortion_parser.add_argument('--random_drop_ratio', default=None, type=float)
distortion_parser.add_argument('--gaussian_blur_r', default=None, type=int)
distortion_parser.add_argument('--gaussian_std', default=None, type=float)
distortion_parser.add_argument('--gaussian_std_fixed', default=None, type=float)
distortion_parser.add_argument('--median_blur_k', default=None, type=float)
distortion_parser.add_argument('--sp_prob_GS', default=None, type=float)
distortion_parser.add_argument('--sp_prob_fixed', default=None, type=float)
distortion_parser.add_argument('--brightness_factor', default=None, type=float)
distortion_parser.add_argument('--contrast_factor', default=None, type=float)
distortion_parser.add_argument('--hue_factor', default=None, type=float)
distortion_parser.add_argument('--saturation_factor', default=None, type=float)
distortion_parser.add_argument('--resize_resolution', default=None, type=int)
distortion_parser.add_argument('--resize_ratio_GS', default=None, type=int)


def distort_images(images: typing.Union[Image.Image, typing.List[Image.Image]],
                   r_degree: float = None,
                   jpeg_ratio: int = None,
                   #jpeg_ratio_GS: int = None,
                   crop_scale_TR: float = None,
                   random_crop_ratio: float = None,
                   random_drop_ratio: float = None,
                   gaussian_blur_r: int = None,
                   gaussian_std: float = None,
                   gaussian_std_fixed: float = None,
                   median_blur_k: int = None,
                   sp_prob_GS: float = None,
                   sp_prob_fixed: float = None,
                   brightness_factor: float = None,
                     contrast_factor: float = None,
                     hue_factor: float = None,
                     saturation_factor: float = None,
                   resize_resolution: int = None,
                   resize_ratio_GS: float = None,
                   _useexact:bool = False,
                   **kwargs
                   ) -> typing.Union[Image.Image, typing.List[Image.Image]]:
    """
    Distort image or list of images    

    @param img: PIL image or list of PIL images
    @param r_degree: float
    @param jpeg_ratio: int
    # @param jpeg_ratio_GS: int
    @param crop_scale_TR: float
    @param random_crop_ratio: float
    @param random_drop_ratio: float
    @param gaussian_blur_r: int
    @param gaussian_std: float
    @param gaussian_std_fixed: float
    @param median_blur_k: int
    @param sp_prob_GS: float
    @param sp_prob_fixed: float
    @param brightness_factor: float
    @param resize_resolution: int
    @param resize_ratio_GS: float

    @return: PIL image or list of PIL images depending on what came in
    """
    if isinstance(images, list):
        was_wrapped = True
    else:
        was_wrapped = False
        images = [images]

    distorted_images = []
    for img in images:

        if not (r_degree is not None or jpeg_ratio is not None or brightness_factor is not None or
                                                contrast_factor is not None or hue_factor is not None or
                                                saturation_factor is not None or
                                                crop_scale_TR is not None or random_crop_ratio is not None or
                                                random_drop_ratio is not None or gaussian_blur_r is not None or
                                                gaussian_std is not None or gaussian_std_fixed is not None or
                                                median_blur_k is not None or sp_prob_GS is not None or
                                                sp_prob_fixed is not None or resize_resolution is not None or
                                                resize_ratio_GS is not None):
            raise ValueError("Input must be a PIL Image when applying distortions.")

        # TR
        if r_degree is not None:
            img = transforms.RandomRotation((r_degree, r_degree))(img)
    
        # TR fixed by author
        if jpeg_ratio is not None:
            device = img.device if hasattr(img, 'device') else torch.device("cpu")
            img = v2.JPEG(jpeg_ratio)((img.cpu() * 255).to(torch.uint8)).to(torch.float32) / 255.0
            img = img.to(device)
            # with tempfile.TemporaryDirectory() as temp_dir:
            #     file = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
            #     img.save(file, quality=jpeg_ratio)
            #     img = Image.open(file)
            #     os.remove(file)

        # GS, obsolete
        #if jpeg_ratio_GS is not None:
        #    img.save(f"tmp_{jpeg_ratio_GS}.jpg", quality=jpeg_ratio_GS)
        #    img = Image.open(f"tmp_{jpeg_ratio_GS}.jpg")
    
        # TR, correct way to do it
        if crop_scale_TR is not None:
            img = transforms.RandomResizedCrop(img.size,
                                               scale=(crop_scale_TR if crop_scale_TR is not None else 1,
                                                      crop_scale_TR if crop_scale_TR is not None else 1),
                                               ratio=(1, 1))(img)
            
        # GS
        if random_crop_ratio is not None:
            # does some black bars which is unrealistic
            #set_random_seed(seed)
            width, height, c = np.array(img).shape
            img = np.array(img)
            new_width = int(width * random_crop_ratio)
            new_height = int(height * random_crop_ratio)
            start_x = np.random.randint(0, width - new_width + 1)
            start_y = np.random.randint(0, height - new_height + 1)
            end_x = start_x + new_width
            end_y = start_y + new_height
            padded_image = np.zeros_like(img)
            padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
            img = Image.fromarray(padded_image)
            
        # GS
        if random_drop_ratio is not None:
            #set_random_seed(seed)
            img = transforms.RandomErasing(1.0, scale=(random_drop_ratio**2, random_drop_ratio**2), ratio=(1, 1))(img)
            # width, height, c = np.array(img).shape
            # img = np.array(img)
            # new_width = int(width * random_drop_ratio)
            # new_height = int(height * random_drop_ratio)
            # start_x = np.random.randint(0, width - new_width + 1)
            # start_y = np.random.randint(0, height - new_height + 1)
            # padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
            # img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
            # img = Image.fromarray(img)

        # GS & TR
        if gaussian_blur_r is not None:
            # img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))
            radius = gaussian_blur_r
            sigma = radius
            kernel_size = int(2 * round(3 * sigma) + 1)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)

        # GS
        if median_blur_k is not None:
            img = img.filter(ImageFilter.MedianFilter(median_blur_k))
    
        # GS & TR
        if gaussian_std is not None:
            # old code does some weird clipping and extreme values
            img_shape = np.array(img).shape
            g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
            g_noise = g_noise.astype(np.uint8)
            img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))
            
        # fixed by author
        if gaussian_std_fixed is not None:
            img = v2.GaussianNoise(0, gaussian_std_fixed)(img)
            # img_tensor = transforms.ToTensor()(img)  # Converts to [0, 1] range, shape: [C, H, W]
            # g_noise = torch.randn_like(img_tensor) * gaussian_std_fixed
            # noisy_img_tensor = torch.clamp(img_tensor + g_noise, 0, 1)
            # img = transforms.ToPILImage()(noisy_img_tensor)

        # GS
        if sp_prob_GS is not None:
            # old code does x1.5 times the noise it is supposed to do
            c,h,w = np.array(img).shape
            prob_zero = sp_prob_GS / 2
            prob_one = 1 - prob_zero
            rdn = np.random.rand(c,h,w)
            img = np.where(rdn > prob_one, np.zeros_like(img), img)
            img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
            img = Image.fromarray(img)

        # fixed by author
        if sp_prob_fixed is not None:
            # image = transforms.ToTensor()(img)  # Converts to [0, 1] range, shape: [C, H, W]
            mask = torch.rand_like(img)
            img = torch.where(mask < (sp_prob_fixed)/2, torch.zeros_like(img), img)
            img = torch.where(mask > 1 - (sp_prob_fixed)/2, torch.ones_like(img), img)
            img = img.clamp(0, 1)
            # img = transforms.ToPILImage()()  # Converts back to PIL Image
            # # This may cause trouble with some numpy version so we only import it here
            # import imgaug.augmenters as iaa

            # img_np = np.array(img)
            # augmenter = iaa.SaltAndPepper(sp_prob_fixed)
            # img_noisy = augmenter(image=img_np)
            # img = Image.fromarray(img_noisy)

        # GS & TR
        if brightness_factor is not None:
            if _useexact:
                brightness_factor = (brightness_factor, brightness_factor)
            img = transforms.ColorJitter(brightness=brightness_factor)(img)

        if contrast_factor is not None:
            if _useexact:
                contrast_factor = (contrast_factor, contrast_factor)
            img = transforms.ColorJitter(contrast=contrast_factor)(img)

        if hue_factor is not None:
            if _useexact:
                hue_factor = (hue_factor, hue_factor)
            img = transforms.ColorJitter(hue=hue_factor)(img)

        if saturation_factor is not None:
            if _useexact:
                saturation_factor = (saturation_factor, saturation_factor)
            img = transforms.ColorJitter(saturation=saturation_factor)(img)

        # by author
        if resize_resolution is not None:
            original_size = img.size
            img = img.resize((resize_resolution, resize_resolution),
                             Image.BILINEAR)
            img = img.resize(original_size, Image.BILINEAR)

        # GS
        if resize_ratio_GS is not None:
            img_shape = np.array(img).shape
            resize_size = int(img_shape[0] * resize_ratio_GS)
            img = transforms.Resize(size=resize_size)(img)
            img = transforms.Resize(size=img_shape[0])(img)

        distorted_images.append(img)
            

    return distorted_images if was_wrapped else distorted_images[0]











BASELINES = [
    "crop_scale_TR=0.999",  # THIS is CLEAN, 999 does nothing to scale the image
    "jpeg_ratio=82", "jpeg_ratio=25",
    "median_blur_k=7",
    "gaussian_std_fixed=0.05",
    "gaussian_std_fixed=0.1",
    "gaussian_blur_r=4",
    "sp_prob_fixed=0.05",
    "brightness_factor=3"
    "contrast_factor=2.",
    "hue_factor=0.1",
    "saturation_factor=1.5",
    "random_crop_ratio=0.6",
    "random_drop_ratio=0.6",
    #"crop_scale_TR=0.75",
    #"crop_scale_TR=0.8",
    "crop_scale_TR=0.9",
    #"crop_scale_TR=0.95",
    #"crop_scale_TR=0.99",
    #"crop_scale_TR=0.999",
    #"gaussian_std_fixed=0.1", "gaussian_std_fixed=0.15", "gaussian_std_fixed=0.2",
    #"gaussian_std=0.1",
    #"gaussian_std=0.05",
    "r_degree=3",
    "r_degree=75",
    #"r_degree=10",
    #"sp_prob_fixed=0.1", "sp_prob_fixed=0.15", "sp_prob_fixed=0.2",
    #"sp_prob_GS=0.05"
    ]


import math

def make_image_grid(images, num_rows, padding=10, bg_color=(255, 255, 255)):
    """
    Arrange PIL images into a grid with a given number of rows.

    Parameters:
        images (list of PIL.Image): list of images to arrange
        num_rows (int): desired number of rows
        padding (int): space between images
        bg_color (tuple): background color (R, G, B)

    Returns:
        PIL.Image: the combined grid image
    """

    # Ensure there is at least one image
    if len(images) == 0:
        raise ValueError("No images provided.")

    num_images = len(images)
    num_cols = math.ceil(num_images / num_rows)

    # Assume all images same size
    img_width, img_height = images[0].size

    grid_width = num_cols * img_width + padding * (num_cols + 1)
    grid_height = num_rows * img_height + padding * (num_rows + 1)

    grid_img = Image.new("RGB", (grid_width, grid_height), color=bg_color)

    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols

        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)

        grid_img.paste(img, (x, y))

    return grid_img



def visualize(imgpath="experiments_v1/gen_wm_v1_50000samples_rar_xl_128clusters_greenfrac0.25_penalty5/images/8.png", 
              distortions="gaussian_std_fixed=0.05,jpeg_ratio=60,gaussian_blur_r=3,sp_prob_fixed=0.05,random_drop_ratio=0.3,brightness_factor=3,contrast_factor=1.5,hue_factor=0.1,saturation_factor=2",
              output="test.png",
                   ):
    """
    Visualize distortions on an image
    @param imgpath: path to the image
    @param r_degree: float
    @param jpeg_ratio: int      
    """
    pilimg = Image.open(imgpath)
    img = transforms.ToTensor()(pilimg)
    distorted_imgs = []
    for distortion in distortions.split(","):
        kwargs = {}
        key, value = distortion.split("=")
        kwargs[key] = float(value) if '.' in value else int(value)
        distorted_img = distort_images(img, _useexact=True, **kwargs)
        distorted_imgs.append(distorted_img)
    distorted_imgs_pil = [transforms.ToPILImage()(i.clamp(0, 1)) for i in distorted_imgs]
    # save images by original filename + distortion name
    make_image_grid([pilimg]+distorted_imgs_pil, len(distorted_imgs_pil), padding=2).save(output)
    print("Saved distorted images to", output)


if __name__ == "__main__":
    fire.Fire(visualize)
    # visualize(imgpath="experiments_v2.2/gen_wm_pp_v2.2_50000samples_GPT-L_c2i_wm_clusters64_penalty5_greenfrac=0.5/images/8.png", distortions="brightness_factor=3,contrast_factor=1.5,hue_factor=0.1,saturation_factor=1.5", output="test.png")
