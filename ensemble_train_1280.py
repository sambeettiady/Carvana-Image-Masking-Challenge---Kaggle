from skimage import filters, io, exposure, color, segmentation, feature, morphology
import numpy as np
import os
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from keras.models import Model,load_model
import keras.backend as K
from keras import callbacks
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from scipy import misc

os.chdir('/home/sambeet/data/kaggle/carvana/')

def ensemble_data_generator(batch_size = 1,dataset = 'train'):
    image_list = os.listdir('resized_data/' + dataset + '/')
    np.random.shuffle(image_list)
    while 1:
        for batch_num in range(len(image_list)//batch_size):
            batch_images = image_list[batch_num*batch_size:((batch_num*batch_size) + (batch_size))]
            X = np.zeros((batch_size, 1280, 1920, 5), dtype=np.float128)
            Y = np.zeros((batch_size, 1280, 1920, 1), dtype=np.float128)
            for i,image_name in zip(range(batch_size),batch_images):
                image_rgb = misc.imread('train/' + image_name)/255.
                image_mask_320 = misc.imread('pred_masks/320/' + image_name,flatten=True)/255.
                image_mask_640 = misc.imread('pred_masks/640/' + image_name,flatten=True)/255.
                image_mask = misc.imread('train_masks/' + image_name.split('.')[0] + '_mask.gif', flatten = True)/255.
                X[i,...,1:1919,:3] = image_rgb
                X[i,...,1:1919,3] = image_mask_320
                X[i,...,1:1919,4] = image_mask_640
                Y[i,...,1:1919,0] = image_mask
                i = i + 1
            yield X, Y

train_data_gen = ensemble_data_generator(1)
val_data_gen = ensemble_data_generator(1,'val')

#Create simple model
inp = Input((1280,1920,5))

conv1 = Conv2D(32, 3, activation='relu', padding='same')(inp)
max1 = MaxPooling2D(2)(conv1)
conv2 = Conv2D(20, 3, activation='relu', padding='same')(max1)
max2 = MaxPooling2D(2)(conv2)
conv3 = Conv2D(20, 3, activation='relu', padding='same')(max2)

deconv3 = Conv2DTranspose(20, 3, strides=4, activation='relu', padding='same')(conv3)
deconv2 = Conv2DTranspose(20, 3, strides=2, activation='relu', padding='same')(conv2)

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
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model.compile(Adam(lr=1e-6), dice_loss, metrics=['accuracy', dice_coef])

#tbcallback = callbacks.TensorBoard(log_dir='carvana_logs/', histogram_freq=1,write_graph=True, write_images=True)

history = model.fit_generator(train_data_gen, epochs = 10, steps_per_epoch= 4088,initial_epoch=9,
                                validation_data = val_data_gen, validation_steps = 512,verbose = 1)

model.save('keras_unet_1280_1920_ensemble_final_9epochs.hd5')
print 'Model Saved!'
model.load_weights('keras_unet_1280_1920_ensemble_final_9epochs.hd5')#,custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

def show_pred_mask(num):
    image_rgb = misc.imread('train/' + image_list[num])/255.
    image_all_mask = misc.imread('train_all_masks/' + image_list[num].split('_')[0] + '.jpg',flatten=True)/255.
    image_bw = misc.imread('train/' + image_list[num],flatten=True)/255.
    angle_id = int(image_list[num].split('.')[0].split('_')[1])
    min_mask = misc.imread('min_and_max_masks_1280/' + 'min_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    max_mask = misc.imread('min_and_max_masks_1280/' + 'max_' + str(int(angle_id) - 1) + '.jpg',flatten=True)/255.
    X = np.zeros((1, 1280, 1920, 12), dtype=np.float32)
    X[0,...,1:1919,:3] = image_rgb
    X[0,...,1:1919,3] = image_all_mask
    X[0,...,1:1919,4] = min_mask
    X[0,...,1:1919,5] = max_mask
    high_contrast = filters.rank.enhance_contrast(image_bw,np.ones((10,10)))
    high_contrast = filters.sobel(high_contrast)
    high_contrast = canny(high_contrast)
    high_contrast = ndi.binary_closing(high_contrast,structure=np.ones((5,5)))
    high_contrast = ndi.binary_fill_holes(high_contrast,np.ones((3,3)))
    X[0,...,1:1919,6] = high_contrast
    normal_contrast = filters.sobel(image_bw)
    normal_contrast = canny(normal_contrast)
    normal_contrast = ndi.binary_closing(normal_contrast,structure=np.ones((5,5)))
    X[0,...,1:1919,7] = ndi.binary_fill_holes(normal_contrast,np.ones((3,3)))
    threshold_otsu_val = filters.threshold_otsu(image_bw)
    threshold_otsu = image_bw > threshold_otsu_val
    X[0,...,1:1919,8] = threshold_otsu
    image_index = np.where(np.zeros((1280,1920)) >= 0)
    df = pd.DataFrame()
    df['l1_dist_y'] = abs((image_index[0] - 639.5)/639.5)
    df['l1_dist_x'] = abs((image_index[1] - 959.5)/959.5)
    df['l2_dist'] = np.sqrt((df['l1_dist_x'])**2 + (df['l1_dist_y'])**2)/np.sqrt(2)
    X[0,...,9] = df.l2_dist.reshape((1280, 1920))
    X[0,...,10] = df.l1_dist_x.reshape((1280, 1920))
    X[0,...,11] = df.l1_dist_y.reshape((1280, 1920))
    pred_mask = model.predict(X).reshape((1280,1920))
    pred_mask[pred_mask >= 0.5] = 1.
    pred_mask[pred_mask < 0.5] = 0.
    pred_mask = ndi.binary_closing(pred_mask,np.ones((1,1))).astype(int)
    pred_mask = ndi.binary_fill_holes(pred_mask,np.ones((10,10))).astype(int)
    pred_mask = morphology.binary_opening(pred_mask,np.ones((10,10)))
    act_mask = misc.imread('train_masks/' + image_list[num].split('.')[0] + '_mask.gif',flatten=True)/255.
    z = skm.confusion_matrix(act_mask.astype(int).flatten(),pred_mask[...,1:1919].astype(int).flatten())
    dice_coeff = 2*(z[1][1])/float(2*z[1][1] + z[0][1] + z[1][0])
    print 'Dice Coeff: ' + str(dice_coeff)
    fig = plt.figure(figsize = (13,13))
    ax1 = fig.add_subplot(121)
    io.imshow(pred_mask[...,1:1919])
    ax2 = fig.add_subplot(221)
    io.imshow(act_mask)
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

#calculate_dice_coeff()
#show_pred_mask(390)
