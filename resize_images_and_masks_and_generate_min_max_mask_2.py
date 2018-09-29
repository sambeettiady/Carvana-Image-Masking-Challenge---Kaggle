from scipy import misc
import numpy as np
import os
import skimage.transform as skt
from skimage.transform import downscale_local_mean

os.chdir('/home/sambeet/data/kaggle/carvana/')

image_list = os.listdir('train/')
car_ids = list(set([image_name.split('_')[0] for image_name in image_list]))
angle_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']

def image_mask_all_angles(car_id):
    all_angles = [car_id + '_0' + str(angle) + '.jpg' if angle <= 9 else car_id + '_' + str(angle) + '.jpg' for angle in range(1,17)]
    min_image = np.full((1280,1918),255.)
    max_image = np.zeros((1280,1918))
    for image in all_angles:
        img = misc.imread('test/' + image,flatten = True).astype(int)
        min_image = np.minimum(min_image,img)
        max_image = np.maximum(max_image,img)
    io.imsave('train_all_masks/' + car_id + '.jpg',(max_image - min_image)/255.)

min_mask = np.ones((16,320,480))
max_mask = np.zeros((16,320,480))

for car_id in car_ids:
    mask_all_angles = misc.imread('train_all_masks/' + car_id + '.jpg')
    mask_all_angles_resized = skt.resize(mask_all_angles/255.,(320,480))
    misc.imsave('train_all_angle_masks_resized/' + car_id + '.jpg',mask_all_angles_resized)
    for angle_id in angle_ids:
        image_rgb = misc.imread('train/' + car_id + '_' + angle_id + '.jpg')
        mask = misc.imread('train_masks/' + car_id + '_' + angle_id + '_mask.gif',flatten=True)
        image_rgb_resized = skt.resize(image_rgb/255.,(320,480))
        mask_resized = skt.resize(mask/255.,(320,480))
        misc.imsave('train_resized/' + car_id + '_' + angle_id + '.jpg',image_rgb_resized)
        misc.imsave('train_masks_resized/' + car_id + '_' + angle_id + '_mask.jpg',mask_resized)
        min_mask[int(angle_id) - 1,...] = np.minimum(min_mask[int(angle_id) - 1,...],mask_resized)
        max_mask[int(angle_id) - 1,...] = np.maximum(max_mask[int(angle_id) - 1,...],mask_resized)

for i in range(0,16):
    misc.imsave(min_mask[i,...],'min_and_max_masks/' + 'min_' + str(i) + '.jpg')
    misc.imsave(max_mask[i,...],'min_and_max_masks/' + 'max_' + str(i) + '.jpg')
