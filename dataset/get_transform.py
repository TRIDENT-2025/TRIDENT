import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import augly.audio as audaugs

IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530] 
IMAGENET_PIXEL_STD = [58.395, 57.12, 57.375]

def salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.01):
    img_array = image.numpy()
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img_array.size * (1. - salt_vs_pepper))

    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0

    return torch.from_numpy(img_array)
    
def get_audio_transform(args, is_training):
    if is_training:
        return lambda waveform, sr: (waveform, sr)  # No augmentations
    else:
        return audaugs.Compose([
            audaugs.AddBackgroundNoise(),
            audaugs.Harmonic(),
            audaugs.OneOf([
                audaugs.PitchShift(),
                audaugs.Clicks(),
                audaugs.ToMono(),
                audaugs.ChangeVolume(volume_db=10.0, p=0.5)
            ]),
        ])

def get_img_transform(args, is_training):

    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_scale = getattr(args, 'test_scale', args.scale_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)

    interpolation = Image.BICUBIC
    if getattr(args, 'interpolation', None) and  args.interpolation == 'bilinear':
        interpolation = Image.BILINEAR 
    
    normalize = get_normalize()

    if is_training:
        ret = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(train_crop_size, interpolation=interpolation),
            ]
        )
    else:
        ret = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(test_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.GaussianBlur(3),
                lambda x: salt_and_pepper_noise(x, salt_vs_pepper=0.45, amount=0.02),
            ]
        )
    return ret

def get_rf_transform(args, is_training, augment='default'):

    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_scale = getattr(args, 'test_scale', args.scale_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)
    interpolation = Image.BICUBIC
    
    if getattr(args, 'interpolation', None) and  args.interpolation == 'bilinear':
        interpolation = Image.BILINEAR 
    
    if is_training:
        ret = transforms.Compose(
            [
                transforms.Resize(train_crop_size, interpolation=interpolation),
                transforms.ToTensor(),
            ]
        )
    else:
        ret = transforms.Compose(
            [
                transforms.Resize(test_crop_size, interpolation=interpolation),
                transforms.ToTensor(),
            ]
        )
    return ret

def get_normalize():
    normalize = transforms.Normalize(
        mean=torch.Tensor(IMAGENET_PIXEL_MEAN) / 255.0,
        std=torch.Tensor(IMAGENET_PIXEL_STD) / 255.0,
    )
    return normalize
