import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

trans = transforms.ToTensor()
img = Image.open("1.jpg").convert('RGB')
mask = Image.open("1.png").convert('L')

img = trans(img)
mask = trans(mask)

x = torch.mul(img, torch.ones_like(mask)-mask)
x = x.transpose(0, 1).transpose(1, 2)
x = x.numpy()
plt.imshow(x)
plt.show()

