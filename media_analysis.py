# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:43:20 2021

@author: mehmet
"""


import numpy as np
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import natsort 

from tensorflow.keras.layers import Dense
# from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

######## positive train data ############

mypath_positive_train=r'C:\Users\mehmet\Desktop\INRIAPerson\Train\pos'

# data=np.genfromtxt(r'C:\Users\mehmet\Desktop\Machine Learning\test_set_01.txt', delimiter='  ', dtype=int)

onlyfiles = [ f for f in listdir(mypath_positive_train) if isfile(join(mypath_positive_train,f)) ]
onlyfiles = (natsort.natsorted(onlyfiles,reverse=False))
images = numpy.empty(len(onlyfiles), dtype=object)

images =[]
for n in range(0,len(onlyfiles)):
   
    image = cv2.imread( join(mypath_positive_train,onlyfiles[n]))
    dim = ( 848,480)

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    # resized = resized.astype('float32')
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imagess = cv2.resize(gray, dsize)
    images.append(resized)
    
images_pos = np.array(images) 
images_pos =images_pos/ 255   
# numpy.save("data.npy",images_pos)
# cv.imshow('Gray image',images[4]) 
label_positive = np.ones(len(images_pos))

########## negative train data ############

mypath_negative_train=r'C:\Users\mehmet\Desktop\INRIAPerson\Train\neg'

# data=np.genfromtxt(r'C:\Users\mehmet\Desktop\Machine Learning\test_set_01.txt', delimiter='  ', dtype=int)

onlyfiles = [ f for f in listdir(mypath_negative_train) if isfile(join(mypath_negative_train,f)) ]
onlyfiles = (natsort.natsorted(onlyfiles,reverse=False))
images = numpy.empty(len(onlyfiles), dtype=object)

images =[]
for n in range(0,len(onlyfiles)):
    
    image = cv2.imread( join(mypath_negative_train,onlyfiles[n]))
    dim = (848, 480)

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    resized =resized/255
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imagess = cv2.resize(gray, dsize)
    images.append(resized)
    
images_neg = np.array(images)   

# cv.imshow('Gray image',images[2]) 
label_negative = np.zeros(len(images_neg))

x_trainn = np.concatenate((images_pos,images_neg))
y_trainn = np.concatenate((label_positive,label_negative))

x_train, y_train = shuffle(x_trainn, y_trainn, random_state=0)

numpy.save("x_train.npy",x_train)


# vggmodel = VGG16(weights='imagenet', include_top=True)

# model = VGG16(weights='imagenet', include_top=True)


baseModel = MobileNetV2(weights="imagenet", include_top=False,
 	input_tensor=Input(shape=(848, 480, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
 	layer.trainable = False
    

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from sklearn.preprocessing import LabelBinarizer

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
        
lenc = MyLabelBinarizer()
y_train =  lenc.fit_transform(y_train)





epochs = 2

history = model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=1)
print(history.history.keys())

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# ###############  Test data ###########

mypath_test_data=r'C:\Users\mehmet\Desktop\INRIAPerson\Train\test data'


onlyfiles = [ f for f in listdir(mypath_test_data) if isfile(join(mypath_test_data,f)) ]
onlyfiles = (natsort.natsorted(onlyfiles,reverse=False))
images = numpy.empty(len(onlyfiles), dtype=object)

test_images =[]
for n in range(0,len(onlyfiles)):
    # dsize = (224, 224)
        
    image = cv2.imread( join(mypath_test_data,onlyfiles[n]))
    # dim = (224, 224)

    # resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imagess = cv2.resize(gray, dsize)
    test_images.append(image)
    
x_test = np.array(test_images)



results = model.predict(test_images)

predicted1 = np.argmax(results,axis=1)
predicted = results[:,1]
# predicted = predicted.tolist()





cv2.waitKey(0)
cv2.destroyAllWindows()