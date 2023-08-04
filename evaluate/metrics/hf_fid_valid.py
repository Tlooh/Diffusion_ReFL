import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from torchmetrics.image.fid import FrechetInceptionDistance

def load_images_from_directory(directory, target_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    image_list = []
    for filename in sorted(os.listdir(directory)):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = transform(image)
        image_list.append(image)

    return torch.stack(image_list)

complex_images = load_images_from_directory("/home/khf/liutao/data/images/complex")
simple_images = load_images_from_directory("/home/khf/liutao/data/images/simple")



fid = FrechetInceptionDistance(normalize=True)
fid.update(complex_images, real=True)
fid.update(simple_images, real=False)

print(f"FID: {float(fid.compute())}")
# FID: 177.7147216796875