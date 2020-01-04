import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math

def image_comp(image, img_size, batch_size, dim):

    row = int(math.sqrt(batch_size))
    col = math.ceil(batch_size // row)
    result = np.ndarray(shape=(dim, img_size * col, img_size * row), dtype=np.float32)
    for i in range(batch_size):
        im = image[i, :].cpu().data.numpy()
        x = i // col
        y = i % row
        result[:, x * img_size : x * img_size + img_size, y * img_size : y * img_size + img_size] = im
    return result

trans = transforms.ToTensor()
img = Image.open("1.jpg").convert('RGB')
mask = Image.open("1.png").convert('RGB')

# img = trans(img)
# mask = trans(mask)

# x = torch.mul(img, torch.ones_like(mask)-mask)
# x = x.transpose(0, 1).transpose(1, 2)
# x = x.numpy()
# plt.imshow(x)
# plt.show()

imgs = []

img = img.resize((256, 256),Image.ANTIALIAS)
x = img.transpose(0, 1).transpose(1, 2)
x = x.numpy()
imgs.append(x)
mask = mask.resize((256, 256),Image.ANTIALIAS)
x = mask.transpose(0, 1).transpose(1, 2)
x = x.numpy()
imgs.append(mask)
y = image_comp(imgs, img_size=256, batch_size=2, dim=3)

