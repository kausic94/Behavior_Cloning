#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 11:59:39 2017

@author: kausic
"""

import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential, Model
from keras.layers.core import Dropout,Flatten,Dense,Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Layer
import time
import matplotlib.pyplot as plt
# Open the CSV file, seperate the data and split into training and test set
log_file_name='driving_log_2'
log_file=open("/home/kausic/My_Projects/Self_Driving_Car/Udacity_SDC/Term1/L14/CarND-Behavioral-Cloning-P3/"+log_file_name+"/driving_log.csv")
reader=csv.reader(log_file)
data=[i for i  in reader]  

log_file.close()
steer_Data_n=np.asarray(data)

correction=.2
# Data Augmentation Step

center_images=steer_Data_n[:,0]
steer_Data_center=np.float32(steer_Data_n[:,3])

left_images=steer_Data_n[:,1]
steer_Data_left=np.float32(steer_Data_n[:,3])+correction

right_images=steer_Data_n[:,2]
steer_Data_right=np.float32(steer_Data_n[:,3])-correction

images_data=np.concatenate((center_images,left_images,right_images))
steer_data =np.concatenate((steer_Data_center,steer_Data_left,steer_Data_right))


zero_data=steer_data[steer_data==0]
zero_images=images_data[steer_data==0]

negative_data=steer_data[steer_data<0]
negative_images=images_data[steer_data<0]

positive_data=steer_data[steer_data>0]
positive_images=images_data[steer_data>0]

balanced_count=min(len(zero_data),len(negative_data),len(positive_data))

to_remove=len(zero_data)-balanced_count
for i in range(to_remove):
    index_removal=np.random.randint(0,len(zero_data))
    zero_data=np.delete(zero_data,index_removal)
    zero_images=np.delete(zero_images,index_removal)

to_remove=len(negative_data)-balanced_count
for i in range(to_remove):
    index_removal=np.random.randint(0,len(negative_data))
    negative_data=np.delete(negative_data,index_removal)
    negative_images=np.delete(negative_images,index_removal)

to_remove=len(positive_data)-balanced_count
for i in range(to_remove):
    index_removal=np.random.randint(0,len(positive_data))
    positive_data=np.delete(positive_data,index_removal)
    positive_images=np.delete(positive_images,index_removal)    

final_steer=np.concatenate((zero_data,negative_data,positive_data))
final_images=np.concatenate((zero_images,negative_images,positive_images))
final_data=np.vstack((final_images,final_steer))
final_data=final_data.transpose()
print(final_data.shape)

np.random.shuffle(final_data)
train_samples, validation_samples = train_test_split(final_data, test_size=0.2)

#Generators for obtaining the data
def get_data(samples,batch_size=128):
    samples_length=len(samples)

    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0,samples_length,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            angles=[]
            for item in batch_samples:
                name=item[0].split('/')[-1]
                name=log_file_name+"/IMG/"+name
                img=cv2.imread(name)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
                images.append(img)
                angles.append(item[1])
            X_train=np.array(images)
            Y_train=np.array(angles)
            Y_train=Y_train.reshape((-1,1))
            yield sklearn.utils.shuffle(X_train,Y_train)

def resizer(i):
    from keras import backend as K
    i=K.tf.image.resize_images(i,[66,200])
    return i
train_generator = get_data(train_samples, batch_size=128)
validation_generator = get_data(validation_samples, batch_size=128)

ch,r,c=3,160,320
model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(r,c,ch)))
model.add(Lambda(resizer))
model.add(Lambda(lambda x : x/255.0 -0.5)) #Normalisation
model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),padding="valid",activation='relu')) #Convolution1
model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=(2,2),padding="valid",activation='relu')) #Convolution2
model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=(2,2),padding="valid",activation='relu')) #Convolution3
model.add(Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="valid",activation='relu')) #Convolution4
model.add(Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="valid",activation='relu')) #Convolution5
model.add(Flatten())
model.add(Dense(100,activation='relu')) #Fully connected layer 1
model.add(Dropout(.25))                 # TO avoid over fitting
model.add(Dense(50,activation='relu')) # Fully connected layer 2
model.add(Dropout(.25))
model.add(Dense(10,activation='relu')) # Fully connected layer 3
model.add(Dropout(.25))
model.add(Dense(1))
t=time.time()
model.compile(loss='mse',optimizer='adam')
history=model.fit_generator(train_generator,samples_per_epoch=len(train_samples),validation_data= validation_generator,nb_val_samples=len(validation_samples),nb_epoch=2,verbose=1)
model.save("model3.h5")
print("total time taken :", time.time()-t)
print (model.summary())
