import os
from PIL import Image
import numpy as np
import pylab as pl
import keras

os.chdir('/home/sambeet/data/kaggle/carvana/train/')

img = Image.open(open('00087a6bd4dc_01.jpg'))
# dimensions are (height, width, channel)
img = np.asarray(img, dtype='float64') / 256.0

img_ = img.transpose(2, 0, 1).reshape(1, 3, 1280, 1918)
pl.imshow(img)




img = imd.imread('00087a6bd4dc_01.jpg')
img_grey = imd.imread('00087a6bd4dc_01.jpg',as_grey=True)
img_mask = imd.imread('../train_masks/00087a6bd4dc_01_mask.gif')

show_image.imshow(img)
show_image.imshow_collection([img,img_grey,img_mask])


