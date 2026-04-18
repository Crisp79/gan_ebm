import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),

        A.Affine(
            translate_percent=0.03,
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),

        A.RandomResizedCrop(
            size=(size, size),   # IMPORTANT FIX
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            p=0.5
        ),

        A.HorizontalFlip(p=0.5),

        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
            p=0.5
        ),

        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),

        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2()
    ])


def get_test_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2()
    ])