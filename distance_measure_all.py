#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:08:03 2020

@author: ivan
"""

 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:48:15 2020

@author: ivan
"""

import os
import numpy as np
import cv2
import csv


def write_csv_features(data,z): #napise 1 red u csv filu, tj 108 brojeva u 1 redu
    with open('features_distance.csv','wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            fieldnames = []
            for i in range(1, len(features[0])+1):
                fieldnames.append(i)
            if z == 0 :
                filewriter.writerow(fieldnames)
                z = 1
            if z:
                filewriter.writerows(data)
    csvfile.close()
    
def write_csv_label(label,z):
     with open('labels_distance.csv','wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            fieldnames = ['angle', 'depth']
            if z == 0:
                filewriter.writerow(fieldnames)
                z = 1
            if z :
                for i in label:
                    filewriter.writerow(i)
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
    
    directory = '/home/ivan/catkin2_workspace/src/neural_networks/src/cnn_degrees/images'
    #image_path = directory+'/*.png'
    
    files = os.listdir(directory)
    files.sort()
    
    
    ##za nulti 
    image0 = cv2.imread(directory+"/edgeprobe_0_0.png") #360x640x3
    old_gray0 = gray_image(directory+"/edgeprobe_0_0.png") #360x640
    _,contours0, hierarchy = cv2.findContours(old_gray0.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black0 = np.zeros((old_gray0.shape[0], old_gray0.shape[1],3), 'uint8')
    centroidi0 = []
   
    for c in contours0:
        cx, cy = get_contour_center(c)
        centroidi0.append([cx,cy])
    centroidi0.sort(key = sort_y) # sad je sortirano po y
    
    sorted0=[]
    k = 0
    
    for l in range(0,12) :
        pom_list = centroidi0[k:k+9] # od 1:10 uzme 9 centorida
        pom_list.sort()
        #print (pom_list)
        k = k+9
        sorted0.append(pom_list)
    
    
    
    
    #sad po ostalim slikama
    
    features = []
    label = []

    for image_name in files:
        filepath = directory + '/' + image_name
        image_name_split = image_name.split('.')[0]
        image_name_split = image_name_split.split('_')
        angle = float(image_name_split[1])
        depth = float(image_name_split[2])
        
        if depth >= 1.0 : #jer na 0mm se ne vidi kut
            image = cv2.imread(filepath) #360x640x3
            gray = gray_image(filepath) #360x640
           
            _,contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            centroidi = []
               
            for c in contours:
                cx, cy = get_contour_center(c)
                centroidi.append([cx,cy])
            
            #sortiranje
            centroidi.sort(key = sort_y) # sad je sortirano po y
        
            sorted_n = []
            j = 0
            for i in range(0,12) :
               pom_list = centroidi[j:j+9] # od 1:10 uzme 9 centorida
               pom_list.sort()
               sorted_n.append(pom_list)
               j= j+9
            
           
               
            razlika = np.asarray(sorted0)-np.asarray(sorted_n)
            
            distance = []
            #da se to spremi u jedan red
            for line in razlika:
                for i in line :
                    [x,y] = i
                    distance.append(x)
                    distance.append(y)
               
            y = [angle,depth]
            label.append(y)
            
            features.append(distance)
        
    #ucitavanje u csv i dodavanje headera, 
    #zato doda jos jednu kolonu napre s rednim brojevima, pa to treba izbrisati kasnije kad se ucita
    write_csv_features(np.float32(features),0)#na kraju 198x108
   
    write_csv_label(label,0) #na kraju 198
  
    
    

       
       
       
       
       
