import torch
from PIL import Image
import cv2
import torchvision.transforms as transforms
import numpy as np
import math
from tensorboardX import SummaryWriter

def image_comp(image, img_size, batch_size, dim):

    col = int(math.sqrt(batch_size))
    row = math.ceil(batch_size // col)
    result = np.ndarray(shape=(dim, img_size * col, img_size * row), dtype=np.uint8)
    for i in range(batch_size):
        im = image[i]
        x = i % col
        y = i // col
        result[:, x * img_size : x * img_size + img_size, y * img_size : y * img_size + img_size] = im
    return result

trans = transforms.ToTensor()
norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img = cv2.imread("1.jpg")
mask = cv2.imread("1.png")

# image test
# img = trans(img)
# mask = trans(mask)
# x = torch.mul(img, torch.ones_like(mask)-mask)
# x = x.transpose(0, 1).transpose(1, 2)
# x = x.numpy()
# plt.imshow(x)
# plt.show()


# image compose
imgs = []
img = cv2.resize(img, (256, 256))
img = np.transpose(img, (2,0,1))
imgs.append(img)
mask = cv2.resize(mask, (256, 256))
mask = np.transpose(mask, (2,0,1))
imgs.append(mask)
y = image_comp(imgs, img_size=256, batch_size=2, dim=3)


# image summary
writer = SummaryWriter("./logs")
# img = np.asarray(img)
# img = trans(img)
#img = np.asarray(img, dtype="float64")
# img = img/255.
# img = np.transpose(img, (2,0,1))
# mask = trans(mask)
# writer.add_image('mask', mask, 1)
# img1 = np.zeros((3, 100, 100))
# img1[0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img1[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

writer.add_image('image', y, 0)
writer.close()