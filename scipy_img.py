from scipy import misc,ndimage
import os
import matplotlib.pyplot as plt

os.chdir('/home/sambeet/data/kaggle/carvana/')

img = misc.imread('train/00087a6bd4dc_01.jpg')
plt.imshow(img)

mask = misc.imread('train_masks/00087a6bd4dc_01_mask.gif',flatten=True)
plt.imshow(mask)

masked_img = img.copy()
masked_img[mask == 0] = 0
plt.imshow(masked_img)

