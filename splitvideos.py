# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:48:21 2021

@author: mehmet
"""


import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('test.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('image'+str(i)+'.jpg',frame)
    i+=1

img = cv2.imread(r'C:\Users\mehmet\Desktop\INRIAPerson\Train\test data\image0.jpg')
cv2.imshow('img',img) 
cap.release()
cv2.destroyAllWindows()