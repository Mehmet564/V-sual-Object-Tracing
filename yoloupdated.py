# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:26:42 2021

@author: mehmet
"""


import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import pylab
import cv2 as cv
import matplotlib.lines
import os
from sklearn.utils import shuffle

import numpy as np
# from medialastversion import predicted
label = ['person']
predicted =np.arange(2,4,0.1)

predicted = predicted.tolist()

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
mypath=r'C:\Users\mehmet\Desktop\INRIAPerson\Train\test data'

# data=np.genfromtxt(r'C:\Users\mehmet\Desktop\Machine Learning\test_set_01.txt', delimiter='  ', dtype=int)

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
import natsort 
onlyfiles = (natsort.natsorted(onlyfiles,reverse=False))

images = numpy.empty(len(onlyfiles), dtype=object)
img =[]

for n in range(0,len(onlyfiles)):
    
    # Loading image
    imgs = cv2.imread(join(mypath,onlyfiles[n]))
    img.append(imgs)
image = img    
for j in range(len(image)): 
    acc = [predicted[j]]
    
    img = cv2.resize(image[j], None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
    
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, acc, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            confidence = acc[i]
            label = str([label])
            color = (0,0,255)
            for (x, y, w, h) in boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label + " " + str(round(predicted[j],3)), (x, y -30), font, 1, color, 2)
    
                
                
            
    cv2.imwrite('img{}'.format(j)+'.jpg',img) 

cv2.imshow("Image", img)
cv2.waitKey()
cv2.destroyAllWindows()