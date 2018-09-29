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

image_name = '0cdf5b5d0ce1_01'
car_id = image_name.split('_')[0]
image = misc.imread('train_sample/' + image_name + '.jpg',flatten = True).astype(float)
image_rgb = misc.imread('train_sample/' + image_name + '.jpg')
image_rgb = image_rgb.flatten()

#meta_file = pd.read_csv('metadata.csv')
angle_indicators = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
angle_indicator = angle_indicators[0]

nnet = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (20,10,5),random_state = 37,
                        verbose = True,early_stopping = False)
image_list = [os.path.basename(x).split('.')[0] for x in glob.glob('train_sample/*' + angle_indicator + '.jpg')]
for image_name, index in zip(image_list,range(1,len(image_list)+1)):
    print image_name,index
    image_rgb = misc.imread('train_sample/' + image_name + '.jpg')
    X = image_rgb.reshape(1,-1)
    image_mask = misc.imread('train_masks/' + image_name + '_mask.gif',flatten=True).astype(int)/255
    Y = image_mask.reshape(1,-1)
    if index == 1:
        nnet.fit(X = X,y = Y)
    else:
        nnet.partial_fit(X = X,y = Y)
joblib.dump(nnet,'nnet_trial1_' + angle_indicator + '.pkl')

image_rgb = misc.imread('test_sample/6cc98271f4dd_01.jpg')
X = image_rgb.reshape(1,-1)
y = nnet.predict(X).astype(int)
y = y.reshape(1280,1918).astype(int)
image_mask = misc.imread('train_masks/' + '6cc98271f4dd_01' + '_mask.gif',flatten=True).astype(int)/255
io.imshow(image_mask - y)
