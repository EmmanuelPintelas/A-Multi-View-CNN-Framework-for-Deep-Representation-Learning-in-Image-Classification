#import efficientnet.keras as efn 
# noisy-student
#base_model = efn.EfficientNetB5(weights='imagenet',   pooling = max, include_top = False, input_shape = (size, size,3) )
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.applications import ResNet50, DenseNet201, Xception, VGG16, InceptionV3, InceptionResNetV2, MobileNetV2, NASNetMobile
import keras
from tensorflow.keras.models import Model
"""
@author: Pintelas Emmanouil
"""
from keras.callbacks import ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from numpy import save, load
# Even more RELIABLE EXPLANATIONS
# AMA KANW MONO CLUSTERING STHN EIKONA, TOTE EKSASFALIZW OTI, OTI TELIKO FEATURE BGALW THA XEI 0 CORELLATION
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import models
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from   sklearn                   import tree
from sklearn.tree import export_text
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
import pandas as pd
import random as rn
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import*
import random
from sklearn import metrics
from keras.models import Sequential
from sklearn.naive_bayes import GaussianNB
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
#from Functions import*
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
import cv2

def Kernel_Tr(kernel, initial_image):
    b_im = initial_image
    b_im_inv = np.copy(b_im)
    if kernel=="Init":
        return  b_im_inv
    elif kernel=="Inv":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = 255-b_im[:,:,0], 255-b_im[:,:,1], 255-b_im[:,:,2] 
        return  b_im_inv
    elif kernel=="x2":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,0]*2, b_im[:,:,1]*2, b_im[:,:,2]*2 
        return  b_im_inv
    elif kernel=="x3":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,0]*3, b_im[:,:,1]*3, b_im[:,:,2]*3 
        return  b_im_inv
    elif kernel=="x5x4x3":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,0]-b_im[:,:,1]-b_im[:,:,2], b_im[:,:,2]-b_im[:,:,0]-b_im[:,:,1], b_im[:,:,1]-b_im[:,:,2]-b_im[:,:,0]
        return  b_im_inv
    elif kernel=="dia_x":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,1]/((0.01)*b_im[:,:,0]), b_im[:,:,2]/((0.01)*b_im[:,:,1]), b_im[:,:,2]/((0.01)*b_im[:,:,0])
        return  b_im_inv
    elif kernel=="dia_x2":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,1]/((0.015)*b_im[:,:,0]), b_im[:,:,2]/((0.015)*b_im[:,:,1]), b_im[:,:,2]/((0.015)*b_im[:,:,0])
        return  b_im_inv
    elif kernel=="_1_dia_x3":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = 1/((0.0003)*b_im[:,:,0]), 1/((0.0003)*b_im[:,:,1]), 1/((0.0003)*b_im[:,:,2])
        return  b_im_inv
    elif kernel=="C_Sqrt":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,2]*b_im[:,:,0], b_im[:,:,0]*b_im[:,:,1], b_im[:,:,1]*b_im[:,:,2]
        return  b_im_inv
    elif kernel=="C_Sqrt2":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = 0.005*b_im[:,:,1]*b_im[:,:,0], 0.005*b_im[:,:,2]*b_im[:,:,1], 0.005*b_im[:,:,0]*b_im[:,:,2]
        return  b_im_inv
    elif kernel=="_try_dia_R10_":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = 1000/( b_im[:,:,2] ), 1000/( b_im[:,:,1] ),  1000/( b_im[:,:,0] )
        return  b_im_inv
    elif kernel=="Square_Normalized":
        a0, a1, a2 = np.square(b_im[:,:,0].astype(int)), np.square(b_im[:,:,1].astype(int)), np.square(b_im[:,:,2].astype(int))
        m0, m1, m2 = np.max(a0), np.max(a1), np.max(a2) 
        d0, d1, d2 = m0/255, m1/255, m2/255
        a0, a1, a2 = a0/d0, a1/d1, a2/d2 
        a0, a1, a2 = a0.astype(int), a1.astype(int), a2.astype(int)
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = a0, a1, a2
        return  b_im_inv
    elif kernel=="Cube_Normalized":
        a0, a1, a2 =  b_im[:,:,0], b_im[:,:,1], b_im[:,:,2]
        a0, a1, a2 = a0.astype(int)**3, a1.astype(int)**3, a2.astype(int)**3
        m0, m1, m2 = np.max(a0), np.max(a1), np.max(a2) 
        d0, d1, d2 = m0/255, m1/255, m2/255
        a0, a1, a2 = a0/d0, a1/d1, a2/d2 
        a0, a1, a2 = a0.astype(int), a1.astype(int), a2.astype(int)
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = a0, a1, a2
        return  b_im_inv


def Kernel_Transformation(kernel, B_IMAGES_Resized):
            B_IMAGES_Tr = []
            for init_image in B_IMAGES_Resized:
                    tr_image = Kernel_Tr(kernel, init_image)
                    B_IMAGES_Tr.append(tr_image)
            return B_IMAGES_Tr

def Standarize (data):
    fff = data
    # # # #---------------------------> Standarize
    for i in range(len(fff[0])):
                fff[:, i] = (fff[:, i] - np.mean(fff[:,i]))/np.std(fff[:,i])
    ii = np.argwhere(np.isnan(fff))
    for i,j in ii:
            fff[i,j] = 0
    data= fff
    return data

def Resize(B_IMAGES, s_ize):
        B_IMAGES_Resized = []
        for b_image in B_IMAGES:
            b_im = cv2.resize(b_image,(s_ize, s_ize)) # Image.resize()
            B_IMAGES_Resized.append(b_im)
        return B_IMAGES_Resized

def CNN_SETUP (model_name, training_images, training_labels):

            train_generator = ImageDataGenerator(rescale = 1./255,
                    rotation_range=10,  
                    zoom_range = 0.1, 
                    width_shift_range=0.1,  
                    height_shift_range=0.1, 
                    shear_range=0.2,
                    horizontal_flip=True,
                    fill_mode="nearest") 



            size = len(training_images[0][0])
            x_train, y_train = training_images, training_labels

            if 1==1:
                base_model = VGG16(include_top=False, weights="imagenet", input_shape=(size, size,3))
                model = Sequential()
                model.add(base_model)
                model.add(GlobalAveragePooling2D())#model.add(Flatten())# 
                model.add(Dense(64,activation='relu'))
                #model.add(Dropout(0.4))
                model.add(Dense(2, activation='softmax'))

                cnt = 0
                for layer in base_model.layers:
                    cnt += 1
                    if isinstance(layer, BatchNormalization):
                        layer.trainable = True
                    else:
                        layer.trainable = False
                for layer in base_model.layers[cnt-int(cnt/5):]:  #
                    layer.trainable = True

                model.compile(optimizer=optimizers.Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
                learn_control = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1, factor=.5, min_lr=0.000005)
                training_set = train_generator.flow(x_train, y_train)
                model.fit_generator(generator=training_set,
                                            steps_per_epoch=len(training_set[0][0]),
                                            verbose=1,
                                            epochs=15,
                                            callbacks=[learn_control])

            return model

train_generator = ImageDataGenerator(rescale = 1./255,
                    rotation_range=10,  
                    zoom_range = 0.1, 
                    width_shift_range=0.1,  
                    height_shift_range=0.1, 
                    shear_range=0.2,
                    horizontal_flip=True,
                    fill_mode="nearest") 
test_generator = ImageDataGenerator(rescale = 1./255)


generator = ImageDataGenerator()
data = generator.flow_from_directory('Dataset',
                                            target_size = (224, 224),
                                            batch_size = 2800,
                                            class_mode = 'categorical',
                                            shuffle=False)


X = data[0][0]
Y = data[0][1]

s3 = np.arange(Y.shape[0]) 
np.random.shuffle(s3)

#### save("s3",s3)
#### s3 = load ("s3.npy")
X = X[s3]
Y = Y[s3]




from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=False, random_state=0)
kf.get_n_splits(X)

Acc_List = []
F1_List = []

if 1 == 1:
        GM_score = []
        Ts_score = []
        Ts_F1_score = []
        sensitivity_score = []
        specificity_score = []
        PPV_score = []
        NPV_score = []
        ROC_AUC_score = []
        for train_index, test_index in kf.split(X):
           x_train, x_test = X[train_index], X[test_index]
           y_train, y_test = Y[train_index], Y[test_index]



           x_train_to_use = np.copy(x_train)
           x_test_to_use = np.copy(x_test)

           model = CNN_SETUP ("Res", x_train_to_use, y_train)  
    #         layer_name = 'dense'
    #         CNN = Model(inputs=model.input,
    #                                                             outputs=model.get_layer(layer_name).output)
    #         CNN.save("Inc_224.h5")

           train_set = train_generator.flow(x_train, y_train)
           _train_x = train_set.x/255
           test_set = test_generator.flow(x_test, y_test)
           _test_x = test_set.x/255

           _test_y = test_set.y[:,1]
           _train_y = train_set.y[:,1]
           pr_ts = model.predict_classes(_test_x) 

           pr_ts = model.predict_classes(_test_x) 
           prob_ts = model.predict_proba(_test_x) [:,1]

           Acc = accuracy_score(_test_y, pr_ts) 
           tn, fp, fn, tp = confusion_matrix(_test_y, pr_ts).ravel()
           GM = (tp*tn)**(0.5)
           ROC_AUC = metrics.roc_auc_score(_test_y, prob_ts)

           print(np.round (Acc, 3))
           print(np.round (GM, 3))
           print(np.round (ROC_AUC, 3)) 

           Ts_score.append (Acc)
           GM_score.append (GM)
           ROC_AUC_score.append (ROC_AUC)



        Acc = np.round (np.mean (np.array(Ts_score)), 3)
        GM  = np.round (np.mean (np.array(GM_score)), 3)
        ROC_AUC  = np.round (np.mean (np.array(ROC_AUC_score)), 3)
        print(np.round (Acc, 3))
        print(np.round (GM, 3))
        print(np.round (ROC_AUC, 3)) 

        S = 1


