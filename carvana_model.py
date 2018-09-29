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
        img = misc.imread('train/' + image,flatten = True).astype(int)
        min_image = np.minimum(min_image,img)
        max_image = np.maximum(max_image,img)
    io.imsave('train_all_masks/' + car_id + '.jpg',(max_image - min_image)/255.)

image_list = os.listdir('train/')
image_list = list(set([image_name.split('_')[0] for image_name in image_list]))

#for car_id in image_list:
#    image_mask_all_angles(car_id)

def image_to_df(image_name):
    car_id = image_name.split('_')[0]
    image = misc.imread('train_sample/' + image_name + '.jpg',flatten = True).astype(float)
    image_rgb = misc.imread('train_sample/' + image_name + '.jpg')
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
    return df
#http://flothesof.github.io/removing-background-scikit-image.html

#meta_file = pd.read_csv('metadata.csv')
angle_indicators = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
angle_indicator = angle_indicators[0]

for angle_indicator in angle_indicators:
    nnet = MLPClassifier(solver = 'adam', activation = 'logistic', hidden_layer_sizes = (20,10,5,2), 
                        random_state = 37,batch_size = 5000,verbose = True)
    image_list = [os.path.basename(x).split('.')[0] for x in glob.glob('train_sample/*' + angle_indicator + '.jpg')]
    for image_name, index in zip(image_list,range(1,len(image_list)+1)):
        print image_name,index
        img_df = image_to_df(image_name)
        X = img_df[[col for col in img_df.columns if col != 'mask']]
        Y = img_df['mask']
        if index == 1:
            nnet.partial_fit(X = X,y = Y,classes = [0,1])
        else:
            nnet.partial_fit(X = X,y = Y)
    joblib.dump(nnet,'nnet_20_10_5_2_adam_logistic_' + angle_indicator + '.pkl')

#print nnet.coefs_
