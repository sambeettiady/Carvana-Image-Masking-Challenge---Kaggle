from skimage import filters, io, exposure, color, segmentation, measure
from scipy import misc
from skimage.feature import canny
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
import math as mth
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import os
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import sklearn.metrics as skm
import matplotlib.pyplot as plt

os.chdir('/home/sambeet/data/kaggle/carvana/')

image_list = os.listdir('train_resized/')

x = misc.imread('train_resized/' + image_list[0],flatten = True)
hist = np.histogram(x, bins=np.arange(0, 256))
plt.plot(hist[1][:-1], hist[0], lw=2)
z = filters.scharr((x/255.))
#z = filters.gaussian(z,1.)
z = canny(z,1)
z = ndi.binary_closing(z,structure=np.ones((3,3)))
z = ndi.binary_fill_holes(z,np.ones((3,3)))
io.imshow(z)
z = filters.prewitt((x-127.5)/255.)
z = filters.scharr((x-127.5)/255.)
x = abs(x - 127.5)/127.5
threshold_li = filters.threshold_yen((x))
threshold_li = filters.threshold_otsu(x)
threshold_li = filters.threshold_isodata(x)
z = x > threshold_li
io.imshow(z)
x = misc.imread('train/' + image_list[0],flatten = True)
x = abs(x - 127.5)/127.5
#x_rgb = misc.imread('train/' + image_list[0])
threshold_li = filters.try_all_threshold(z1)
z = x > threshold_li
io.imshow((z -.5))
z1 = filters.rank.autolevel((x-127.5)/127.5,disk(20))
