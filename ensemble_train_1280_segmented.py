import numpy as np
import os
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from keras.models import Model,load_model
import keras.backend as K
from scipy import misc

os.chdir('/home/sambeet/data/kaggle/carvana/')

def ensemble_data_generator(dataset = 'train'):
    image_list = os.listdir('resized_data/' + dataset + '/')
    np.random.shuffle(image_list)
    while 1:
        for image in image_list:
            X = np.zeros((1, 1280, 1920, 5), dtype=np.float128)
            Y = np.zeros((1, 1280, 1920, 1), dtype=np.float128)
            image_rgb = misc.imread('train/' + image)/255.
            image_mask_320 = misc.imread('pred_masks/320/' + image,flatten=True)/255.
            image_mask_640 = misc.imread('pred_masks/640/' + image,flatten=True)/255.
            image_mask = misc.imread('train_masks/' + image.split('.')[0] + '_mask.gif', flatten = True)/255.
            X[0,...,1:1919,:3] = image_rgb
            X[0,...,1:1919,3] = image_mask_320
            X[0,...,1:1919,4] = image_mask_640
            Y[0,...,1:1919,0] = image_mask
            for i in range(0,4):
                Z_x = np.zeros((1, 640, 960, 5), dtype=np.float128)
                Z_y = np.zeros((1, 640, 960, 1), dtype=np.float128)
                if i == 0:
                    Z_x[0,...] = X[0,0:640,0:960,...]
                    Z_y[0,...] = Y[0,0:640,0:960,...]
                elif i == 1:
                    Z_x[0,...] = X[0,0:640,960:1920,...]
                    Z_y[0,...] = Y[0,0:640,960:1920,...]
                elif i == 2:
                    Z_x[0,...] = X[0,640:1280,0:960,...]
                    Z_y[0,...] = Y[0,640:1280,0:960,...]
                else:
                    Z_x[0,...] = X[0,640:1280,960:1920,...]
                    Z_y[0,...] = Y[0,640:1280,960:1920,...]
                yield Z_x, Z_y

train_data_gen = ensemble_data_generator('train')
val_data_gen = ensemble_data_generator('val')

#Create simple model
inp = Input((640,960,5))

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
conv6 = Conv2D(32, 3, activation='relu', padding='same')(max5)

deconv6 = Conv2DTranspose(32, 3, strides=32, activation='relu', padding='same')(conv6)
deconv5 = Conv2DTranspose(32, 3, strides=16, activation='relu', padding='same')(conv52)
deconv4 = Conv2DTranspose(32, 3, strides=8, activation='relu', padding='same')(conv42)
deconv3 = Conv2DTranspose(32, 3, strides=4, activation='relu', padding='same')(conv32)
deconv2 = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(conv22)

deconvs = concatenate([conv12, deconv2, deconv3, deconv4, deconv5,deconv6])

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

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model.compile(Adam(lr=1e-7), dice_loss, metrics=['accuracy', dice_coef])

#tbcallback = callbacks.TensorBoard(log_dir='carvana_logs/', histogram_freq=1,write_graph=True, write_images=True)
model.load_weights('keras_unet_1280_1920_ensemble_segmented_14epochs.hd5')#,custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

history = model.fit_generator(train_data_gen, epochs = 16, steps_per_epoch= 16352,initial_epoch=14,
                                validation_data = val_data_gen, validation_steps = 2048,verbose = 1)

model.save('keras_unet_1280_1920_ensemble_segmented_16epochs.hd5')
print 'Model Saved!'
