#Load packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy import misc
import skimage.transform as skt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model,load_model
from sklearn.preprocessing import LabelEncoder
import keras.backend as K

#Change working directory
os.chdir('/home/sambeet/data/hackerearth/deep learning challenge 2/')

#Load csv data and create test, train and validation datasets
all_data = pd.read_csv('csv/train.csv')
all_data = pd.get_dummies(data=all_data,drop_first=True,columns=['gender'])
all_data['age'][all_data.age < 2] = 2
all_data['age'][all_data.age > 95] = 95
all_data['age'] = (all_data.age - 2)/93.
#all_data.drop(labels=['view_position'],axis=1,inplace=True)
train_data, X_test = train_test_split(all_data, test_size = 3729, random_state = 37)
test_data, val_data = train_test_split(X_test, test_size = 0.5, random_state = 37)
#train_data.to_csv('csv/train_data.csv')
#test_data.to_csv('csv/test_data.csv')
#val_data.to_csv('csv/val_data.csv')

#Resize images into 256 and 512
#for image in all_data.image_name:
#    image_1024 = misc.imread('train/train_/' + image)
#    image_512 = skt.resize(image_1024,(512,512))
#    misc.imsave('train_resized/512/' + image,image_512)
#    image_256 = skt.resize(image_1024,(256,256))
#    misc.imsave('train_resized/256/' + image,image_256)

lb_test_data = pd.read_csv('csv/test.csv')
lb_test_data = pd.get_dummies(data=lb_test_data,drop_first=True,columns=['gender'])
lb_test_data['age'][lb_test_data.age < 2] = 2
lb_test_data['age'][lb_test_data.age > 95] = 95
lb_test_data['age'] = (lb_test_data.age - 2)/93.
#lb_test_data.drop(labels=['view_position'],axis=1,inplace=True)
lb_test_data.to_csv('csv/lb_test_data.csv')

#Resize test images into 512
#for image in lb_test_data.image_name:
#    image_1024 = misc.imread('test/test_/' + image)
#    image_512 = skt.resize(image_1024,(512,512))
#    misc.imsave('test_resized/512/' + image,image_512)

#EDA
#all_data.describe(include='all')
#sns.distplot(train_data.age)
#all_data = pd.get_dummies(data=all_data,drop_first=True,columns=['gender'])
#all_data = pd.get_dummies(data=all_data,drop_first=False,columns=['detected'])
#all_data.groupby(['detected']).mean()
#Check for image intensity, mean/median and histogram difference
#x = train_data.copy()
#columns = ['class_' + str(num) for num in range(1,15)]
#y = pd.DataFrame(columns=columns)
#for disease in columns:
#    intensity_dist = [np.std(misc.imread('train/train_/' + image)/255.) for image in train_data.image_name[train_data.detected == disease][0:48]]
#    y[disease] = intensity_dist
#sns.boxplot(y)
#np.histogram(misc.imread('train/train_/' + image)/255., bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9,1])[0][0:8]/float(1024*1024)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_data.detected.values)
encoded_Y = encoder.transform(train_data.detected.values)

K.clear_session()
model = load_model('logs/resnet50_1e-5.47-2.29.hdf5')

#
def model_prediction_lb(resolution = 512):
    batch_size = 25
    df = lb_test_data.copy()
    last_index = lb_test_data.shape[0]
    image_list = list(df.image_name.values)
    numeric_variables = df[['age','gender_M','view_position']].values
    predictions = []
    iters = np.int32(np.ceil(len(image_list)/float(batch_size)))
    for batch_num in range(iters):
        start_index = batch_num*batch_size
        end_index = (batch_num + 1)*batch_size
        if end_index > last_index:
            end_index = last_index
        batch_images = image_list[start_index:end_index]
        images = np.empty((end_index - start_index, resolution, resolution, 3), dtype = np.float32)
        numeric_data_1 = numeric_variables[start_index:end_index]
        numeric_data_2 = np.empty((end_index - start_index,8), dtype = np.float32)
        for i,image_name in zip(range(end_index - start_index),batch_images):
                ima_copy = misc.imread('test_resized/' + str(resolution) + '/' + image_name,flatten = True)/255.
                images[i,...,0] = ima_copy
                images[i,...,1] = ima_copy
                images[i,...,2] = ima_copy
                numeric_data_2[i,...] = np.histogram(images[i,...,0], bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9,1])[0][0:8]/float(resolution*resolution)
        numeric_data = np.hstack((numeric_data_1,numeric_data_2))
        batch_predict = model.predict_on_batch([images,numeric_data])
        batch_predict = batch_predict.argmax(1)
        batch_predict = encoder.inverse_transform(batch_predict)
        for prediction in batch_predict:
            predictions.append(prediction)
    return predictions

x = model_prediction_lb()
lb_test_data['detected'] = x
lb_test_data[['row_id','detected']].to_csv('submission_17.csv',index = False)
