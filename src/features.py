
from torchvision import transforms
from PIL import Image

def geo_safe_augmentations():
    """
    Return a torchvision transform suitable for geospatial tiles:
    rotation small angles, flips, color jitter moderate, random crop/zoom mild.
    """
    aug = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),   # small rotation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # vertical flip may be ok for tiles but lower prob
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)], p=0.5),
        transforms.RandomResizedCrop(size=(64,64), scale=(0.9,1.0), ratio=(0.95,1.05)),
        transforms.ToTensor(),
    ])
    return aug

def basic_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
