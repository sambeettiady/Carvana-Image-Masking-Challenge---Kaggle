from skimage import filters, io, exposure, color, segmentation, feature, morphology
import skimage
import glob
from scipy import misc
from skimage.feature import canny
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import os
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from keras.models import Model,load_model
import keras.backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
import skimage.transform as skt

os.chdir('/home/sambeet/data/kaggle/carvana/')

smooth = 1.
# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model_320 = load_model('keras_unet_320_480_final_25epochs.hd5',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})
model_640 = load_model('keras_unet_640_960_final_20epochs.hd5',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})
#model_1280 = load_model('keras_unet_1280_1920.hd5',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

image_list_train = os.listdir('resized_data/train/')
image_list_val = os.listdir('resized_data/val/')
image_list_test = os.listdir('resized_data/test/')

def predict_and_save(image_list,image_type = 'train'):
    for i,image in zip(range(1,len(image_list) + 1),image_list):
        #####Resolution 320*480 prediction#####        
        image_rgb_320 = misc.imread('resized_data/' + image_type + '/' + image)/255.
        image_all_mask_320 = misc.imread('train_all_angle_masks_resized/' + image.split('_')[0] + '.jpg',flatten=True)/255.
        image_bw_320 = misc.imread('resized_data/' + image_type + '/' + image,flatten=True)/255.
        angle_id = int(image.split('.')[0].split('_')[1])
        min_mask_320 = misc.imread('min_and_max_masks/' + 'min_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
        max_mask_320 = misc.imread('min_and_max_masks/' + 'max_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
        X_320 = np.empty((1, 320, 480, 12), dtype=np.float32)
        X_320[0,...,:3] = image_rgb_320
        X_320[0,...,3] = image_all_mask_320
        X_320[0,...,4] = min_mask_320
        X_320[0,...,5] = max_mask_320
        high_contrast_320 = filters.rank.enhance_contrast(image_bw_320,np.ones((10,10)))
        high_contrast_320 = filters.sobel(high_contrast_320)
        high_contrast_320 = canny(high_contrast_320)
        high_contrast_320 = ndi.binary_closing(high_contrast_320,structure=np.ones((5,5)))
        high_contrast_320 = ndi.binary_fill_holes(high_contrast_320,np.ones((3,3)))
        X_320[0,...,6] = high_contrast_320
        normal_contrast_320 = filters.sobel(image_bw_320)
        normal_contrast_320 = canny(normal_contrast_320)
        normal_contrast_320 = ndi.binary_closing(normal_contrast_320,structure=np.ones((5,5)))
        X_320[0,...,7] = ndi.binary_fill_holes(normal_contrast_320,np.ones((3,3)))
        threshold_otsu_val_320 = filters.threshold_otsu(image_bw_320)
        threshold_otsu_320 = image_bw_320 > threshold_otsu_val_320
        X_320[0,...,8] = threshold_otsu_320
        image_index_320 = np.where(image_bw_320 >= 0)
        df_320 = pd.DataFrame()
        df_320['l1_dist_y'] = abs((image_index_320[0] - 159.5)/159.5)
        df_320['l1_dist_x'] = abs((image_index_320[1] - 239.5)/239.5)
        df_320['l2_dist'] = np.sqrt((df_320['l1_dist_x'])**2 + (df_320['l1_dist_y'])**2)/np.sqrt(2)
        X_320[0,...,9] = df_320.l2_dist.reshape((320,480))
        X_320[0,...,10] = df_320.l1_dist_x.reshape((320,480))
        X_320[0,...,11] = df_320.l1_dist_y.reshape((320,480))
        pred_mask_320 = model_320.predict(X_320).reshape((320,480))
        pred_mask_320 = skt.resize(pred_mask_320,(1280,1918))
        pred_mask_320[pred_mask_320 >= 0.5] = 1.
        pred_mask_320[pred_mask_320 < 0.5] = 0.
        pred_mask_320 = ndi.binary_closing(pred_mask_320,np.ones((1,1))).astype(int)
        pred_mask_320 = ndi.binary_fill_holes(pred_mask_320,np.ones((10,10))).astype(int)
        pred_mask_320 = morphology.binary_opening(pred_mask_320,np.ones((10,10)))
        misc.imsave('pred_masks/320/' + image,pred_mask_320)
        
        del pred_mask_320,X_320,df_320
        
        #####Resolution 640*960 prediction#####
        image_rgb_640 = misc.imread('train_resized_640/' + image)/255.
        image_all_mask_640 = misc.imread('train_all_angle_masks_resized_640/' + image.split('_')[0] + '.jpg',flatten=True)/255.
        image_bw_640 = misc.imread('train_resized_640/' + image,flatten=True)/255.
        angle_id = int(image.split('.')[0].split('_')[1])
        min_mask_640 = misc.imread('min_and_max_masks_640/' + 'min_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
        max_mask_640 = misc.imread('min_and_max_masks_640/' + 'max_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
        X_640 = np.empty((1, 640, 960, 12), dtype=np.float32)
        X_640[0,...,:3] = image_rgb_640
        X_640[0,...,3] = image_all_mask_640
        X_640[0,...,4] = min_mask_640
        X_640[0,...,5] = max_mask_640
        high_contrast_640 = filters.rank.enhance_contrast(image_bw_640,np.ones((10,10)))
        high_contrast_640 = filters.sobel(high_contrast_640)
        high_contrast_640 = canny(high_contrast_640)
        high_contrast_640 = ndi.binary_closing(high_contrast_640,structure=np.ones((5,5)))
        high_contrast_640 = ndi.binary_fill_holes(high_contrast_640,np.ones((3,3)))
        X_640[0,...,6] = high_contrast_640
        normal_contrast_640 = filters.sobel(image_bw_640)
        normal_contrast_640 = canny(normal_contrast_640)
        normal_contrast_640 = ndi.binary_closing(normal_contrast_640,structure=np.ones((5,5)))
        X_640[0,...,7] = ndi.binary_fill_holes(normal_contrast_640,np.ones((3,3)))
        threshold_otsu_val_640 = filters.threshold_otsu(image_bw_640)
        threshold_otsu_640 = image_bw_640 > threshold_otsu_val_640
        X_640[0,...,8] = threshold_otsu_640
        image_index_640 = np.where(image_bw_640 >= 0)
        df_640 = pd.DataFrame()
        df_640['l1_dist_y'] = abs((image_index_640[0] - 319.5)/319.5)
        df_640['l1_dist_x'] = abs((image_index_640[1] - 479.5)/479.5)
        df_640['l2_dist'] = np.sqrt((df_640['l1_dist_x'])**2 + (df_640['l1_dist_y'])**2)/np.sqrt(2)
        X_640[0,...,9] = df_640.l2_dist.reshape((640,960))
        X_640[0,...,10] = df_640.l1_dist_x.reshape((640,960))
        X_640[0,...,11] = df_640.l1_dist_y.reshape((640,960))
        pred_mask_640 = model_640.predict(X_640).reshape((640,960))
        pred_mask_640 = skt.resize(pred_mask_640,(1280,1918))
        pred_mask_640[pred_mask_640 >= 0.5] = 1.
        pred_mask_640[pred_mask_640 < 0.5] = 0.
        pred_mask_640 = ndi.binary_closing(pred_mask_640,np.ones((1,1))).astype(int)
        pred_mask_640 = ndi.binary_fill_holes(pred_mask_640,np.ones((10,10))).astype(int)
        pred_mask_640 = morphology.binary_opening(pred_mask_640,np.ones((10,10)))
        misc.imsave('pred_masks/640/' + image,pred_mask_640)
        
        del pred_mask_640,X_640,df_640

        print 'Progress: ' + str(100*i/float(len(image_list))) + '%'

predict_and_save(image_list=image_list_test,image_type='test')
predict_and_save(image_list=image_list_val,image_type='val')
predict_and_save(image_list=image_list_train,image_type='train')
