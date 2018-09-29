from skimage import filters, io, exposure, color
import glob
from scipy import misc
from skimage.feature import canny
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import os
from sklearn.externals import joblib
import csv

os.chdir('/home/sambeet/data/kaggle/carvana/')

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_submission(image_name):
    car_id = image_name.split('_')[0]
    angle_indicator = image_name.split('_')[1]
    nnet_model = joblib.load('nnet_20_10_5_2_adam_logistic_' + angle_indicator +'.pkl')
    image = misc.imread('test/' + image_name + '.jpg',flatten = True).astype(float)
    image_rgb = misc.imread('test/' + image_name + '.jpg')
    image_float = image_rgb.astype(float)
    image_mask_all_angles = misc.imread('train_all_masks/' + car_id + '.jpg',flatten = True)/255.
    image_index = np.where(image >= 0)
    sobel = filters.sobel(image)
    sobel_blurred = filters.gaussian(sobel,sigma=1)
    canny_filter_image = canny(image/255.)
    threshold_li = filters.threshold_li(image)
    mask_li = image > threshold_li
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    laplace = filters.laplace(image)
    threshold_local_51 = filters.threshold_local(image,51)
    mask_local_51 = image > threshold_local_51
    df = pd.DataFrame()
    df['l1_dist_y'] = abs((image_index[0] - 639.5)/639.5)
    df['l1_dist_x'] = abs((image_index[1] - 958.5)/958.5)
    df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
    df['grey_values'] = image.reshape((1,1918*1280))[0]/255.
    df['red_values'] = image_rgb.reshape((3,1918*1280))[0]/255.
    df['blue_values'] = image_rgb.reshape((3,1918*1280))[1]/255.
    df['green_values'] = image_rgb.reshape((3,1918*1280))[2]/255.
    df['red_float'] = image_float.reshape((3,1918*1280))[0]/255.
    df['blue_float'] = image_float.reshape((3,1918*1280))[1]/255.
    df['green_float'] = image_float.reshape((3,1918*1280))[2]/255.
    df['sobel_blurred'] = sobel_blurred.reshape((1,1918*1280))[0]/255.
    df['canny_filter_image'] = canny_filter_image.reshape((1,1918*1280))[0].astype(int)
    df['sobel_h'] = sobel_h.reshape((1,1918*1280))[0]/255.
    df['sobel_v'] = sobel_v.reshape((1,1918*1280))[0]/255.
    df['laplace'] = laplace.reshape((1,1918*1280))[0]/511.
    df['threshold_local_51'] = mask_local_51.reshape((1,1918*1280))[0].astype(int)
    df['threshold_li'] = mask_li.reshape((1,1918*1280))[0].astype(int)
    df['pred_mask'] = nnet_model.predict(X = df[[col for col in df.columns if col != 'mask']])
    pred_mask = np.array(df['pred_mask']).astype(float)
    pred_mask = pred_mask.reshape((1280,1918))
    rle = rle_encode(pred_mask)
    rle_str = rle_to_string(rle)
    return rle_str

image_list = os.listdir('test/')

with open('submission.csv','wb') as sub_file:
    row_writer = csv.writer(sub_file)
    row_writer.writerow(['img','rle_mask'])
    for img in image_list:
        image_name = img.split('.')[0]
        rle_test = rle_submission(image_name)
        row_writer.writerow([img,rle_test])
