from skimage import filters, io, exposure, color
import glob
from scipy import misc
from skimage.feature import canny
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import os
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import sklearn.metrics as skm

os.chdir('/home/sambeet/data/kaggle/carvana/')

def image_mask_all_angles(car_id):
    all_angles = [car_id + '_0' + str(angle) + '.jpg' if angle <= 9 else car_id + '_' + str(angle) + '.jpg' for angle in range(1,17)]
    min_image = np.full((1280,1918),255.)
    max_image = np.zeros((1280,1918))
    for image in all_angles:
        img = misc.imread('test/' + image,flatten = True).astype(int)
        min_image = np.minimum(min_image,img)
        max_image = np.maximum(max_image,img)
    io.imsave('train_all_masks/' + car_id + '.jpg',(max_image - min_image)/255.)

image_list = os.listdir('test/')
image_list = list(set([image_name.split('_')[0] for image_name in image_list]))

for car_id in image_list:
    image_mask_all_angles(car_id)
