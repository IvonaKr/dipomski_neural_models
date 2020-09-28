#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:46:07 2020

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor


dataset_x = pd.read_csv('features.csv')
dataset_y = pd.read_csv('labels.csv')
X = dataset_x.iloc[:,:].values 
y = dataset_y.iloc[:,:].values



y_angle = y[:,0]
y_depth =  y[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size = 0.2,
                                                    random_state = 0)


X_train = X
X_test = X_test
y_train = y
y_test = y_test

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler() #moraju biti 2 objekta
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#pazi da y bude matrica! 
y_train= sc_y.fit_transform(y_train.reshape(-1,2))
y_test = sc_y.transform(y_test.reshape(-1,2))





from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
#regressor.fit(X,y)
y_pred= MultiOutputRegressor(regressor).fit(X_train,y_train).predict(X_test)
#y_pred= MultiOutputRegressor(regressor).fit(X,y).predict(X)

#jedan_x_test = X_test[0]
#jedan_y_test = y_test[0] 

#y_pred = regressor.predict(sc_X.transform(np.array([jedan_x_test])))
#y_pred = sc_y.inverse_transform(y_pred)

#y_pred = regressor.predict(X_test)


#plt.plot(y, y_pred, '.')

#vizualizacija 

y_pred_true = sc_y.inverse_transform(y_pred)
y_test_true = sc_y.inverse_transform(y_test)

angle_test = y_test_true[:,0]
angle_pred = y_pred_true[:,0]
angle_test, angle_pred = (list(t) for t in zip(*sorted(zip(angle_test, angle_pred))))
depth_test = y_test_true[:,1]
depth_pred = y_pred_true[:,1]
depth_test, depth_pred = (list(t) for t in zip(*sorted(zip(depth_test, depth_pred))))


plt.figure(figsize = (8,8))

plt.subplot(211)
plt.title('SVM multioutput (kernel = linear), features = pin_position')
plt.plot(angle_test, label = 'angle_test')
plt.plot(angle_pred, label = 'angle_pred')
plt.xlabel('picture')
plt.ylabel('angle')
plt.legend(loc = 'lower right')


plt.subplot(212)
plt.plot(depth_test, label = 'depth_test')
plt.plot(depth_pred, label = 'depth_pred')
plt.xlabel('picture')
plt.ylabel('depth')
plt.legend(loc = 'lower right')

plt.show()
#plt.subplot(211)
#plt.plot(y_test[:,0],y_test[:,1],'bo')
#plt.xlabel('angle')
#plt.ylabel('depth')
#
##plt.subplot(212)
#plt.plot(y_pred[:,0],y_pred[:,1],'ro')
#plt.xlabel('angle')
#plt.ylabel('depth')



