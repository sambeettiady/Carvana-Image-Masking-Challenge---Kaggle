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

def test_acc(image_name):
    car_id = image_name.split('_')[0]
    angle_indicator = image_name.split('_')[1]
    nnet_model = joblib.load('nnet_20_10_5_2_adam_logistic_' + angle_indicator +'.pkl')
    image = misc.imread('test_sample/' + image_name + '.jpg',flatten = True).astype(float)
    image_rgb = misc.imread('test_sample/' + image_name + '.jpg')
    image_float = image_rgb.astype(float)
    image_mask_all_angles = misc.imread('train_all_masks/' + car_id + '.jpg',flatten = True)/255.
    image_mask = misc.imread('train_masks/' + image_name + '_mask.gif',flatten = True)/255
#io.imshow(image_mask)
    image_index = np.where(image >= 0)
    sobel = filters.sobel(image)   # working
#io.imshow(sobel)
    sobel_blurred = filters.gaussian(sobel,sigma=1)  # Working
#io.imshow(sobel_blurred)
    canny_filter_image = canny(image/255.)
#io.imshow(canny_filter_image)
#    threshold_niblack_11 = filters.threshold_niblack(sobel_blurred,201)
#io.imshow(threshold_niblack)
    threshold_li = filters.threshold_li(image)
    mask_li = image > threshold_li
#io.imshow(mask)
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    laplace = filters.laplace(image)
    threshold_local_51 = filters.threshold_local(image,51)
    mask_local_51 = image > threshold_local_51
#io.imshow(mask)
    df = pd.DataFrame()
#    df['y'] = (image_index[0] - 639.5)/639.5
#    df['x'] = (image_index[1] - 958.5)/958.5
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
#    df['threshold_niblack_11'] = threshold_niblack_11.reshape((1,1918*1280))[0]#/255.
    df['threshold_li'] = mask_li.reshape((1,1918*1280))[0].astype(int)
    df['mask'] = image_mask.reshape((1,1918*1280))[0]
    df['mask'] = df['mask'].astype('category')
    df['pred_mask'] = nnet_model.predict(X = df[[col for col in df.columns if col != 'mask']])
    z = skm.confusion_matrix(df['mask'],df['pred_mask'])
    dice_coeff = 2*(z[1][1])/float(2*z[1][1] + z[0][1] + z[1][0])
    print 'Dice Coefficient:', dice_coeff
    accuracy = 100*(z[0][0] + z[1][1])/float(sum(sum(z)))
    print 'Accuracy:', accuracy
    precision = 100*(z[1][1])/float(z[0][1] + z[1][1])
    print 'Precision:', precision
    recall = 100*(z[1][1])/float(z[1][0]+z[1][1])
    print 'Recall:', recall
    act_mask = np.array(df['mask'])
    act_mask = act_mask.reshape((1280,1918))
    pred_mask = np.array(df['pred_mask']).astype(float)
    pred_mask = pred_mask.reshape((1280,1918))
    return df,pred_mask, image, act_mask

test1 = test_acc(image_name='0cdf5b5d0ce1_05')
