#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:59:18 2020

@author: ivan
"""
import numpy as np
import cv2
import csv
import glob

path = '/home/ivan/catkin2_workspace/src/neural_networks/src/cnn_degrees/images/'
def write_csv_features(data): #napise 1 red u csv filu, tj 108 brojeva u 1 redu
    with open('features.csv','wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            filewriter.writerows(data)
    csvfile.close()
    
def write_csv_label(label):
     with open('degrees.csv','wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            for i in label:
                filewriter.writerow([i])
     csvfile.close()

def gray_image(path):
    gray_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
    return gray_image

def get_contour_center(contour):
    M = cv2.moments(contour)
    cx = -1
    cy = -1
    if (M['m00'] != 0):
        cx = int(M['m10']/M['m00']) #prema definiciji 
        cy = int(M['m01']/M['m00'])
    return cx,cy

def sort_y(val):
   return val[1]


if __name__=='__main__':
    image0 = cv2.imread(path+"edgeprobe_0_0.png") #360x640x3
    old_gray0 = gray_image(path+"edgeprobe_0_0.png") #360x640
    #slika je crno bila pa ne treba u hsv bla bla
    _,contours0, hierarchy = cv2.findContours(old_gray0.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #u contorurs ih stavi po najvecem y, onda po najvecem x, znaci ne trebam sortirati po y
    black0 = np.zeros((old_gray0.shape[0], old_gray0.shape[1],3), 'uint8')
    
    centroidi0 = []
   
    for c in contours0:
        cx, cy = get_contour_center(c)
        centroidi0.append([cx,cy])
    #sortirat centoride, prvo po y, pa hardcodirano po x
    # po x ih je 9, po y 12
    centroidi0.sort(key = sort_y) # sad je sortirano po y
    
    sorted0 = []
    j = 0
    for i in range(0,12) :
        pom_list = centroidi0[j:j+9] # od 1:10 uzme 9 centorida
        pom_list.sort()
        #print (pom_list)
        sorted0.append(pom_list)
        j= j+9
        
        
       
        
    sorted0 = np.asarray(sorted0) #12x9x2
    sorted0 = sorted0.reshape(-1,2)  #108x2, supac
    
    image1 = cv2.imread(path+"edgeprobe_0_1.png") #360x640x3
    old_gray1 = gray_image(path+"edgeprobe_0_1.png") #360x640
    #slika je crno bila pa ne treba u hsv bla bla
    _,contours1, hierarchy = cv2.findContours(old_gray1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #u contorurs ih stavi po najvecem y, onda po najvecem x, znaci ne trebam sortirati po y
    black1 = np.zeros((old_gray1.shape[0], old_gray1.shape[1],3), 'uint8')
    
    centroidi1 = []
   
    for c in contours1:
        cx, cy = get_contour_center(c)
        centroidi1.append([cx,cy])
    #sortirat centoride, prvo po y, pa hardcodirano po x
    # po x ih je 9, po y 12
    centroidi1.sort(key = sort_y) # sad je sortirano po y
    
    sorted1 = []
    j = 0
    for i in range(0,12) :
        pom_list = centroidi1[j:j+9] # od 1:10 uzme 9 centorida
        pom_list.sort()
       # print (pom_list)
        sorted1.append(pom_list)
        j= j+9
        
        
       
        
    sorted1 = np.asarray(sorted0) #12x9x2
    sorted1 = sorted0.reshape(-1,2)  #108x2, supac

