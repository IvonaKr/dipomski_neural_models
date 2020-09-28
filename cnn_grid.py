#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:50:38 2020

@author: ivan
"""

import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#nova neuronska s prociscenim slikama

## dataset
dim = (64,64)
directory = '/home/ivan/diplomski_ws/src/neural_networks/src/grid/images_grid'
files = os.listdir(directory)

images = []
labels = []

for image_name in files : 
   filepath = directory + '/' + image_name #uzime prvo 0,100,10, a sprema ih normalno
   #print(filepath)
   image_name_split = image_name.split('.')[0]
   image_name_split = image_name_split.split('_')
   angle = float(image_name_split[1])
   depth = float(image_name_split[2]) 
   
   if depth >= 1.0 :
       image = cv2.imread(filepath) #360x640x3
       image = cv2.resize(image,dim) #64x64x3
       image = sc_X.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
       #image = cv2.resize(image,dim)
       images.append(image)
       labels.append([angle,depth])
      
       
images = np.array(images) # ? x (64x64x3)
labels = np.asarray(labels) #? x2
angle = labels[:,1]
angle = angle.reshape(len(angle),-1)

#dataset
X_train, X_test, y_train, y_test = train_test_split(images,labels, 
                                                    test_size = 0.2,
                                                    random_state = 0)
X_train = X_train
y_train = y_train
X_test = X_test
y_test = y_test

#skaliranje, x se gore jer scaliranje moze samo <= 2dim

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


#CNN
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3), activation ='relu')) 
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

#fully connected sloj
model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))  #ovako je za klasifikaciju
model.add(Dense(units = y_train.shape[1], activation = 'linear')) 

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse','mae'])


model.fit(X_train,y_train, epochs = 100, batch_size = 12, verbose = 0)
model.summary()


y_pred = model.predict(X_test)
print("y1 MSE:%.4f" % mean_squared_error(y_test[:,0], y_pred[:,0]))
print("y2 MSE:%.4f" % mean_squared_error(y_test[:,1], y_pred[:,1]))

#y1 mse : 0.1146
#y2 mse : 0.0070

#visualize
plt.figure(figsize=(8,8))

x_ax = range(len(X_test))

plt.subplot(211)
plt.title("Fully connected CNN Multiple Output Regression Grid")
plt.scatter(x_ax, y_test[:,0],s = 6,c ='r', label="angle_test") #s je debljina tocke
plt.plot(x_ax, y_pred[:,0],'g', label="angle_pred")
plt.xlabel('picture')
plt.ylabel('angle')
plt.legend()

plt.subplot(212)
plt.scatter(x_ax, y_test[:,1],  s=6, c = 'r', label="depth_test")
plt.plot(x_ax, y_pred[:,1],'g', label="depth_pred")
plt.xlabel('picture')
plt.ylabel('depth')
plt.legend()
plt.show()

### neskalirani podaci

y_test_true = sc_y.inverse_transform(y_test)
y_pred_true = sc_y.inverse_transform(y_pred)

plt.figure(figsize=(8,8))

plt.subplot(211)
plt.title("Fully connected CNN Multiple Output Regression Grid")
plt.scatter(x_ax, y_test_true[:,0],s = 6,c ='r', label="angle_test") #s je debljina tocke
plt.plot(x_ax, y_pred_true[:,0],'g:.', label="angle_pred")
plt.xlabel('picture')
plt.ylabel('angle')
plt.legend()

plt.subplot(212)
plt.scatter(x_ax, y_test_true[:,1],  s=12, c = 'r', label="depth_test")
plt.plot(x_ax, y_pred_true[:,1],'g:.', label="depth_pred")
plt.xlabel('picture')
plt.ylabel('depth')
plt.legend()
plt.show()

model.save("cnn_regression_grid.h5")

from keras.models import load_model 

model = load_model('cnn_regression_grid.h5')