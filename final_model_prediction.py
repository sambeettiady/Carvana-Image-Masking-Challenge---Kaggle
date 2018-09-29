import numpy as np
import os
from keras.models import Model,load_model
import keras.backend as K
from scipy import misc
from scipy import ndimage as ndi
from skimage import io,morphology,filters
from skimage.feature import canny
from sklearn import metrics

os.chdir('/home/sambeet/data/kaggle/carvana/')

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

final_ensemble = load_model('keras_unet_1280_1920_ensemble_segmented_16epochs.hd5',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})
final_640 = load_model('keras_unet_640_960_final_20epochs.hd5',custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})
X = np.empty((1, 640, 960, 12), dtype=np.float128)
image_name = '0d1a9caf4350_07.jpg'
i = 0
image_rgb = misc.imread('train_resized_640/' + image_name)/255.
image_bw = misc.imread('train_resized_640/' + image_name,flatten=True)/255.
image_all_mask = misc.imread('train_all_angle_masks_resized_640/' + image_name.split('_')[0] + '.jpg', flatten=True)/255.
min_mask = misc.imread('min_and_max_masks_640/' + 'min_6.jpg',flatten=True)/255.
max_mask = misc.imread('min_and_max_masks_640/' + 'max_6.jpg',flatten=True)/255.
X[0,...,:3] = image_rgb
X[0,...,3] = image_all_mask
X[0,...,4] = min_mask
X[0,...,5] = max_mask
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

get_1st_layer_output = K.function([final_640.layers[0].input],[final_640.layers[1].output])
layer_output = get_1st_layer_output([X])[0]
image_list = os.listdir('resized_data/val/')
io.imshow(layer_output[0,...,0])
misc.imsave(arr=layer_output[0,...,61],name='layer_61.png')
#48,61,56,54,43,40,39,32,30,28,14,9,1
for i,image in zip(range(1,len(image_list) + 1),image_list):
    X = np.zeros((1, 1280, 1920, 5), dtype=np.float128)
    pred_mask = np.zeros((1280, 1920), dtype=np.float128)
    image_rgb = misc.imread('train/' + image)/255.
    image_mask_320 = misc.imread('pred_masks/320/' + image,flatten=True)/255.
    image_mask_640 = misc.imread('pred_masks/640/' + image,flatten=True)/255.
    X[0,...,1:1919,:3] = image_rgb
    X[0,...,1:1919,3] = image_mask_320
    X[0,...,1:1919,4] = image_mask_640
    pred_mask[0:640,0:960] = final_ensemble.predict(X[0:,0:640,0:960,...]).reshape((640,960))
    pred_mask[0:640,960:1920] = final_ensemble.predict(X[0:,0:640,960:1920,...]).reshape((640,960))
    pred_mask[640:1280,0:960] = final_ensemble.predict(X[0:,640:1280,0:960,...]).reshape((640,960))
    pred_mask[640:1280,960:1920] = final_ensemble.predict(X[0:,640:1280,960:1920,...]).reshape((640,960))
    pred_mask = pred_mask[...,1:1919]
    pred_mask[pred_mask >= 0.5] = 1.
    pred_mask[pred_mask < 0.5] = 0.
    pred_mask = pred_mask.astype(int)
    misc.imsave('pred_masks/1280/' + image,pred_mask)
    print 'Progress: ' + str(100*i/float(len(image_list))) + '%'

image_list = os.listdir('resized_data/val/')
recall_list = []
precision_list = []
acc_list = []

for i,image in zip(range(1,len(image_list) + 1),image_list):
    pred_mask_1280 = misc.imread('pred_masks/1280/' + image,flatten=True)/255.
    pred_mask_1280[pred_mask_1280 >= 0.5] = 1.
    pred_mask_1280[pred_mask_1280 < 0.5] = 0.
    actual_mask = misc.imread('train_masks/' + image.split('.')[0] + '_mask.gif',flatten=True)/255.
    acc = metrics.accuracy_score(actual_mask.flatten(),pred_mask_1280.astype(int).flatten())
    prec = metrics.precision_score(actual_mask.flatten(),pred_mask_1280.astype(int).flatten())
    rec = metrics.recall_score(actual_mask.flatten(),pred_mask_1280.astype(int).flatten())
    acc_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    print 'Progress: ' + str(100*i/float(len(image_list))) + '%'

#print 'Avg. Dice Coeff: ' + str(np.mean(dice_coef_list_320))
#print 'Avg. Dice Coeff: ' + str(np.mean(dice_coef_list_640))
#print 'Avg. Dice Coeff: ' + str(np.mean(dice_coef_list_1280))

print 'Avg. Recall: ' + str(np.mean(recall_list))
print 'Avg. Precision: ' + str(np.mean(precision_list))
print 'Avg. Accuracy: ' + str(np.mean(acc_list))
