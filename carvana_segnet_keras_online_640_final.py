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

def data_generator(batch_size = 4,dataset = 'train'):
    image_list = os.listdir('resized_data/' + dataset + '/')
    np.random.shuffle(image_list)
    min_mask = np.ones((16,640,960))
    max_mask = np.zeros((16,640,960))
    for angle_id in range(0,16):
        min_mask[angle_id] = misc.imread('min_and_max_masks_640/' + 'min_' + str(angle_id) + '.jpg',flatten=True)/255.
        max_mask[angle_id] = misc.imread('min_and_max_masks_640/' + 'max_' + str(angle_id) + '.jpg',flatten=True)/255.
    while 1:
        for batch_num in range(len(image_list)//batch_size):
            batch_images = image_list[batch_num*batch_size:(batch_num*batch_size)+(batch_size)]
            X = np.empty((batch_size, 640, 960, 12), dtype=np.float128)
            Y = np.empty((batch_size, 640, 960, 1), dtype=np.float128)
            for i,image_name in zip(range(batch_size),batch_images):
                image_rgb = misc.imread('train_resized_640/' + image_name)/255.
                image_bw = misc.imread('train_resized_640/' + image_name,flatten=True)/255.
                image_all_mask = misc.imread('train_all_angle_masks_resized_640/' + image_name.split('_')[0] + '.jpg', flatten=True)/255.
                image_mask = misc.imread('train_masks_resized_640/' + image_name.split('.')[0] + '_mask.jpg', flatten = True)/255.
                image_mask = np.reshape(image_mask,(640,960,1))
                X[i,...,:3] = image_rgb
                X[i,...,3] = image_all_mask
                X[i,...,4] = min_mask[int(image_name.split('.')[0].split('_')[1]) - 1]
                X[i,...,5] = max_mask[int(image_name.split('.')[0].split('_')[1]) - 1]
                high_contrast = filters.rank.enhance_contrast(image_bw,np.ones((10,10)))
                high_contrast = filters.sobel(high_contrast)
                high_contrast = canny(high_contrast)
                high_contrast = ndi.binary_closing(high_contrast,structure=np.ones((5,5)))
                high_contrast = ndi.binary_fill_holes(high_contrast,np.ones((3,3)))
                X[i,...,6] = high_contrast
                normal_contrast = filters.sobel(image_bw)
                normal_contrast = canny(normal_contrast)
                normal_contrast = ndi.binary_closing(normal_contrast,structure=np.ones((5,5)))
                X[i,...,7] = ndi.binary_fill_holes(normal_contrast,np.ones((3,3)))
                threshold_otsu_val = filters.threshold_otsu(image_bw)
                threshold_otsu = image_bw > threshold_otsu_val
                X[i,...,8] = threshold_otsu
                image_index = np.where(image_bw >= 0)
                df = pd.DataFrame()
                df['l1_dist_y'] = abs((image_index[0] - 319.5)/319.5)
                df['l1_dist_x'] = abs((image_index[1] - 479.5)/479.5)
                df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
                X[i,...,9] = df.l2_dist.reshape((640,960))
                X[i,...,10] = df.l1_dist_x.reshape((640,960))
                X[i,...,11] = df.l1_dist_y.reshape((640,960))
                Y[i,...] = image_mask
                i = i + 1
            yield X, Y

train_data_gen = data_generator(1,'train')
val_data_gen = data_generator(1,'val')

#Create simple model
inp = Input((640,960,12))

conv11 = Conv2D(64, 3, activation='relu', padding='same')(inp)
conv12 = Conv2D(64, 3, activation='relu', padding='same')(conv11)
max1 = MaxPooling2D(2)(conv12)
conv21 = Conv2D(32, 3, activation='relu', padding='same')(max1)
conv22 = Conv2D(32, 3, activation='relu', padding='same')(conv21)
max2 = MaxPooling2D(2)(conv22)
conv31 = Conv2D(32, 3, activation='relu', padding='same')(max2)
conv32 = Conv2D(32, 3, activation='relu', padding='same')(conv31)
max3 = MaxPooling2D(2)(conv32)
conv41 = Conv2D(32, 3, activation='relu', padding='same')(max3)
conv42 = Conv2D(32, 3, activation='relu', padding='same')(conv41)
max4 = MaxPooling2D(2)(conv42)
conv51 = Conv2D(32, 3, activation='relu', padding='same')(max4)
conv52 = Conv2D(32, 3, activation='relu', padding='same')(conv51)
max5 = MaxPooling2D(2)(conv52)
conv61 = Conv2D(32, 3, activation='relu', padding='same')(max5)
conv62 = Conv2D(32, 3, activation='relu', padding='same')(conv61)
max6 = MaxPooling2D(2)(conv62)
conv7 = Conv2D(32, 3, activation='relu', padding='same')(max6)

deconv7 = Conv2DTranspose(32, 3, strides=64, activation='relu', padding='same')(conv7)
deconv6 = Conv2DTranspose(32, 3, strides=32, activation='relu', padding='same')(conv62)
deconv5 = Conv2DTranspose(32, 3, strides=16, activation='relu', padding='same')(conv52)
deconv4 = Conv2DTranspose(32, 3, strides=8, activation='relu', padding='same')(conv42)
deconv3 = Conv2DTranspose(32, 3, strides=4, activation='relu', padding='same')(conv32)
deconv2 = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(conv22)

deconvs = concatenate([conv12, deconv2, deconv3, deconv4, deconv5,deconv6,deconv7])

out = Conv2D(1, 7, activation='sigmoid', padding='same')(deconvs)
model = Model(inp, out)
model.summary()

smooth = 1.
# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model.compile(Adam(lr=1e-7), dice_loss, metrics=['accuracy', dice_coef])

#tbcallback = callbacks.TensorBoard(log_dir='carvana_logs/', histogram_freq=1,write_graph=True, write_images=True)

history = model.fit_generator(train_data_gen, epochs = 21, steps_per_epoch= 4088,initial_epoch = 20,
                                validation_data = val_data_gen, validation_steps = 512,verbose = 1)

model.save('keras_unet_640_960_final_20epochs.hd5')

model.load_weights('keras_unet_640_960_final_20epochs.hd5')#,custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

def show_pred_mask(num):
    image_rgb = misc.imread('train_resized_640/' + image_list[num])/255.
    image_all_mask = misc.imread('train_all_angle_masks_resized_640/' + image_list[num].split('_')[0] + '.jpg',flatten=True)/255.
    image_bw = misc.imread('train_resized_640/' + image_list[num],flatten=True)/255.
    angle_id = int(image_list[num].split('.')[0].split('_')[1])
    min_mask = misc.imread('min_and_max_masks_640/' + 'min_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    max_mask = misc.imread('min_and_max_masks_640/' + 'max_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    X = np.empty((1, 640, 960, 12), dtype=np.float32)
    X[0,...,:3] = image_rgb
    X[0,...,3] = image_all_mask
    X[0,...,4] = min_mask
    X[0,...,5] = max_mask
    high_contrast = filters.rank.enhance_contrast(image_bw,np.ones((10,10)))
    high_contrast = filters.sobel(high_contrast)
    high_contrast = canny(high_contrast)
    high_contrast = ndi.binary_closing(high_contrast,structure=np.ones((5,5)))
    high_contrast = ndi.binary_fill_holes(high_contrast,np.ones((3,3)))
    X[0,...,6] = high_contrast
    normal_contrast = filters.sobel(image_bw)
    normal_contrast = canny(normal_contrast)
    normal_contrast = ndi.binary_closing(normal_contrast,structure=np.ones((5,5)))
    X[0,...,7] = ndi.binary_fill_holes(normal_contrast,np.ones((3,3)))
    threshold_otsu_val = filters.threshold_otsu(image_bw)
    threshold_otsu = image_bw > threshold_otsu_val
    X[0,...,8] = threshold_otsu
    image_index = np.where(image_bw >= 0)
    df = pd.DataFrame()
    df['l1_dist_y'] = abs((image_index[0] - 319.5)/319.5)
    df['l1_dist_x'] = abs((image_index[1] - 479.5)/479.5)
    df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
    X[0,...,9] = df.l2_dist.reshape((640,960))
    X[0,...,10] = df.l1_dist_x.reshape((640,960))
    X[0,...,11] = df.l1_dist_y.reshape((640,960))
    pred_mask = model.predict(X).reshape((640,960))
    pred_mask = skt.resize(pred_mask,(1280,1918))
    pred_mask[pred_mask >= 0.5] = 1.
    pred_mask[pred_mask < 0.5] = 0.
    pred_mask = ndi.binary_closing(pred_mask,np.ones((1,1))).astype(int)
    pred_mask = ndi.binary_fill_holes(pred_mask,np.ones((10,10))).astype(int)
    pred_mask = morphology.binary_opening(pred_mask,np.ones((10,10)))
    act_mask = misc.imread('train_masks/' + image_list[num].split('.')[0] + '_mask.gif',flatten=True)/255.
    z = skm.confusion_matrix(act_mask.astype(int).flatten(),pred_mask.astype(int).flatten())
    dice_coeff = 2*(z[1][1])/float(2*z[1][1] + z[0][1] + z[1][0])
    print 'Dice Coeff: ' + str(dice_coeff)
    fig = plt.figure(figsize = (13,13))
    ax1 = fig.add_subplot(121)
    io.imshow(act_mask - pred_mask)
    ax2 = fig.add_subplot(221)
    io.imshow(image_rgb)
    plt.show()

def calculate_dice_coeff(ground_truth=Y_val,test_data=X_val,model=model):
    predicted_masks = model.predict(test_data)
    dice_coef_list = []
    for i in range(predicted_masks.shape[0]):
        pred_mask = predicted_masks[i].reshape((320,480))        
        pred_mask[pred_mask >= 0.5] = 1.
        pred_mask[pred_mask < 0.5] = 0.
        pred_mask = ndi.binary_closing(pred_mask,np.ones((1,1))).astype(int)
        pred_mask = ndi.binary_fill_holes(pred_mask,np.ones((10,10))).astype(int)
        pred_mask = morphology.binary_opening(pred_mask,np.ones((10,10)))
        z = skm.confusion_matrix(ground_truth[i].reshape((320,480)).astype(int).flatten(),pred_mask.astype(int).flatten())
        dice_coef_list.append(2*(z[1][1])/float(2*z[1][1] + z[0][1] + z[1][0]))
    print 'Avg. Dice Coeff: ' + str(np.mean(dice_coef_list))

calculate_dice_coeff()
show_pred_mask(0)
