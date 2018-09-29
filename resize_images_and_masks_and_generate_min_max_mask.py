from scipy import misc
import numpy as np
import os
import skimage.transform as skt

os.chdir('/home/sambeet/data/kaggle/carvana/')

image_list = os.listdir('train/')
car_ids = list(set([image_name.split('_')[0] for image_name in image_list]))
angle_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']

min_mask = np.ones((16,640,960))
max_mask = np.zeros((16,640,960))

for car_id in car_ids:
    mask_all_angles = misc.imread('train_all_masks/' + car_id + '.jpg')
    mask_all_angles_resized = skt.resize(mask_all_angles/255.,(640,960))
    misc.imsave('train_all_angle_masks_resized_640/' + car_id + '.jpg',mask_all_angles_resized)
    for angle_id in angle_ids:
        image_rgb = misc.imread('train/' + car_id + '_' + angle_id + '.jpg')
        mask = misc.imread('train_masks/' + car_id + '_' + angle_id + '_mask.gif',flatten=True)
        image_rgb_resized = skt.resize(image_rgb/255.,(640,960))
        mask_resized = skt.resize(mask/255.,(640,960))
        misc.imsave('train_resized_640/' + car_id + '_' + angle_id + '.jpg',image_rgb_resized)
        misc.imsave('train_masks_resized_640/' + car_id + '_' + angle_id + '_mask.jpg',mask_resized)
        min_mask[int(angle_id) - 1,...] = np.minimum(min_mask[int(angle_id) - 1,...],mask_resized)
        max_mask[int(angle_id) - 1,...] = np.maximum(max_mask[int(angle_id) - 1,...],mask_resized)

'''
for car_id in car_ids:
    for angle_id in angle_ids:
        mask = misc.imread('train_masks_resized/' + car_id + '_' + angle_id + '_mask.jpg',flatten=True)
        min_mask[int(angle_id) - 1,...] = np.minimum(min_mask[int(angle_id) - 1,...],mask)
        max_mask[int(angle_id) - 1,...] = np.maximum(max_mask[int(angle_id) - 1,...],mask)
'''

for i in range(0,16):
    misc.imsave('min_and_max_masks_640/' + 'min_' + str(i + 1) + '.jpg',min_mask[i])
    misc.imsave('min_and_max_masks_640/' + 'max_' + str(i + 1) + '.jpg',max_mask[i])
