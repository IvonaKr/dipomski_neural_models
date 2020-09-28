#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:17:55 2020

@author: ivan
"""


import numpy as np
import cv2
import csv

path = '/home/ivan/catkin2_workspace/src/neural_networks/src/cnn_degrees/images/'


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

def write_csv_features(data): #napise 1 red u csv filu, tj 108 brojeva u 1 redu
    with open('features_distance.csv','wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            filewriter.writerow(data)
    csvfile.close()
    
def write_csv_label(label):
     with open('label_distance.csv','wb+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            filewriter.writerow([label])
     csvfile.close()

def sort_y(val):
   return val[1]

if __name__=='__main__':
    image0 = cv2.imread(path+"edgeprobe_0_0.png") #360x640x3
    old_gray0 = gray_image(path+"edgeprobe_0_0.png") #360x640
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
    
    sorted0=[]
    k = 0
    
    for l in range(0,12) :
        pom_list = centroidi0[k:k+9] # od 1:10 uzme 9 centorida
        pom_list.sort()
        #print (pom_list)
        k = k+9
        sorted0.append(pom_list)
        
        #TO JE TO NE TREBA TI NISTA VISE; IMAS MATRICU
       
            
        
        ## za drugu sliku 
    image1 = cv2.imread(path+"edgeprobe_40_8.png") #360x640x3
    old_gray1 = gray_image(path+"edgeprobe_40_8.png") #360x640
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
    
    sorted1=[]
    k = 0
    
    for l in range(0,12) :
        pom_list = centroidi1[k:k+9] # od 1:10 uzme 9 centorida
        pom_list.sort()
        #print (pom_list)
        k = k+9
        sorted1.append(pom_list)
        
    razlika = np.asarray(sorted0)-np.asarray(sorted1)        
     #dobro su poredani, provjerila sam ručno, samo nije napisano ko matrica
    
    distance = []
    for line in razlika:
        for i in line :
            [x,y] = i
            distance.append(x)
            distance.append(y)
    
        
    write_csv_features(np.float32(distance))
        
        
        
       
        
   # sorted0 = np.asarray(sorted0) #12x9x2
    #sorted0 = sorted0.reshape(-1,2)  #108x2, supac
    

        
  
   # for i,t in enumerate(sorted0):
   #     [cx,cy] = t
   #     cv2.circle(image0, (cx, cy), 2, (0, 0, 255), -1)
   #     cv2.putText(image0, str(i), (cx - 15, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0),1,cv2.LINE_AA)

   
    
    
   
    
   

    
    
    
    
    
    
    
    
    
    #cv2.imshow('image0', image0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()