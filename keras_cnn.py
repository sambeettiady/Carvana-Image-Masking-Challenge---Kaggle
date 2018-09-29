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
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt

os.chdir('/home/sambeet/data/kaggle/carvana/')

image_list = os.listdir('resized_data/train/')
car_ids = list(set([image.split('_')[0] for image in image_list]))
num_cars_train = 10
car_ids_sliced = car_ids[:num_cars_train]

car_ids_train,car_ids_test = train_test_split(car_ids_sliced, test_size=0.1, random_state=37)

#Load dataset
def load_dataset(car_ids_list):
    min_mask = np.ones((16,320,480))
    max_mask = np.zeros((16,320,480))
    for angle_id in range(0,16):
        min_mask[angle_id] = misc.imread('min_and_max_masks/' + 'min_' + str(angle_id) + '.jpg',flatten=True)/255.
        max_mask[angle_id] = misc.imread('min_and_max_masks/' + 'max_' + str(angle_id) + '.jpg',flatten=True)/255.
    i = 0
    X = np.empty((len(car_ids_list)*16, 320, 480, 12), dtype=np.float32)
    Y = np.empty((len(car_ids_list)*16, 320, 480, 1), dtype=np.float32)
    for car_id in car_ids_list:
        for angle in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']:
            image_rgb = misc.imread('resized_data/train/' + car_id + '_' + angle + '.jpg')/255.
            image_bw = misc.imread('resized_data/train/' + car_id + '_' + angle + '.jpg',flatten=True)/255.
            image_all_mask = misc.imread('train_all_angle_masks_resized/' + car_id + '.jpg',flatten=True)/255.
            image_mask = misc.imread('train_masks_resized/' + car_id + '_' + angle + '_mask.jpg',flatten=True).astype(int)/255.
            image_mask = np.reshape(image_mask,(320,480,1))
            X[i,...,:3] = image_rgb
            X[i,...,3] = image_all_mask
            X[i,...,4] = min_mask[int(angle) - 1]
            X[i,...,5] = max_mask[int(angle) - 1]
            z = filters.sobel(image_bw)
            smoothed_edges = filters.gaussian(z,1)
            X[i,...,6] = smoothed_edges
            z = canny(z)
            z = ndi.binary_closing(z,structure=np.ones((5,5)))
            X[i,...,7] = ndi.binary_fill_holes(z,np.ones((3,3))).astype(int)
            threshold_otsu_val = filters.threshold_otsu(image_bw)
            threshold_otsu = image_bw > threshold_otsu_val
            X[i,...,8] = threshold_otsu
            image_index = np.where(z >= 0)
            df = pd.DataFrame()
            df['l1_dist_y'] = abs((image_index[0] - 159.5)/159.5)
            df['l1_dist_x'] = abs((image_index[1] - 239.5)/239.5)
            df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
            X[i,...,9] = df.l2_dist.reshape((320,480))
            X[i,...,10] = df.l1_dist_x.reshape((320,480))
            X[i,...,11] = df.l1_dist_y.reshape((320,480))
            Y[i,...] = image_mask
            i = i + 1
    return X,Y

X_train,Y_train = load_dataset(car_ids_train)
X_test,Y_test = load_dataset(car_ids_test)

#Create simple model
inp = Input((320,480,12))

conv1 = Conv2D(12, 3, activation='relu', padding='same')(inp)
max1 = MaxPooling2D(2)(conv1)
conv2 = Conv2D(12, 3, activation='relu', padding='same')(max1)
max2 = MaxPooling2D(2)(conv2)
conv3 = Conv2D(12, 3, activation='relu', padding='same')(max2)

deconv3 = Conv2DTranspose(12, 3, strides=4, activation='relu', padding='same')(conv3)
deconv2 = Conv2DTranspose(12, 3, strides=2, activation='relu', padding='same')(conv2)

deconvs = concatenate([conv1, deconv2, deconv3])

out = Conv2D(1, 7, activation='sigmoid', padding='same')(deconvs)

model = Model(inp, out)
model.summary()

smooth = 1.
# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model.compile(Adam(lr=1e-3), dice_loss, metrics=['accuracy', dice_coef])

history = model.fit(X_train, Y_train, epochs = 15, validation_data=(X_test, Y_test), batch_size=20, verbose=1)

model.save('fifth_cnn')

model = load_model('fourth_cnn',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})
def show_pred_mask(num):
    image_rgb = misc.imread('train_resized/' + image_list[num])/255.
    image_all_mask = misc.imread('train_all_angle_masks_resized/' + image_list[num].split('_')[0] + '.jpg',flatten=True)/255.
    image_bw = misc.imread('train_resized/' + image_list[num],flatten=True)/255.
    angle_id = int(image_list[num].split('.')[0].split('_')[1])
    min_mask = misc.imread('min_and_max_masks/' + 'min_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    max_mask = misc.imread('min_and_max_masks/' + 'max_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    X = np.empty((1, 320, 480, 12), dtype=np.float32)
    X[0,...,:3] = image_rgb
    X[0,...,3] = image_all_mask
    X[0,...,4] = min_mask
    X[0,...,5] = max_mask
    z = filters.sobel(image_bw)
    smoothed_edges = filters.gaussian(z,1)
    X[0,...,6] = smoothed_edges
    z = canny(z)
    z = ndi.binary_closing(z,structure=np.ones((5,5)))
    X[0,...,7] = ndi.binary_fill_holes(z,np.ones((3,3))).astype(int)
    threshold_otsu_val = filters.threshold_otsu(image_bw)
    threshold_otsu = image_bw > threshold_otsu_val
    X[0,...,8] = threshold_otsu
    image_index = np.where(z >= 0)
    df = pd.DataFrame()
    df['l1_dist_y'] = abs((image_index[0] - 159.5)/159.5)
    df['l1_dist_x'] = abs((image_index[1] - 239.5)/239.5)
    df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
    X[0,...,9] = df.l2_dist.reshape((320,480))
    X[0,...,10] = df.l1_dist_x.reshape((320,480))
    X[0,...,11] = df.l1_dist_y.reshape((320,480))
    pred_mask = model.predict(X).reshape((320,480))
    pred_mask[pred_mask >= 0.5] = 1.
    pred_mask[pred_mask < 0.5] = 0.
    pred_mask = ndi.binary_closing(pred_mask,np.ones((1,1))).astype(int)
    pred_mask = ndi.binary_fill_holes(pred_mask,np.ones((10,10))).astype(int)
    pred_mask = morphology.binary_opening(pred_mask,np.ones((10,10)))
    act_mask = misc.imread('train_masks_resized/' + image_list[num].split('.')[0] + '_mask.jpg',flatten=True)/255.
    z = skm.confusion_matrix(act_mask.astype(int).flatten(),pred_mask.astype(int).flatten())
    dice_coeff = 2*(z[1][1])/float(2*z[1][1] + z[0][1] + z[1][0])
    print 'Dice Coeff: ' + str(dice_coeff)
    fig = plt.figure(figsize = (13,13))
    ax1 = fig.add_subplot(121)
    io.imshow(act_mask - pred_mask)
    ax2 = fig.add_subplot(221)
    io.imshow(image_rgb)
    plt.show()

show_pred_mask(1600)
