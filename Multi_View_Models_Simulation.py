import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from   sklearn                   import tree
from sklearn.tree import export_text
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from scipy.stats import entropy
from scipy.stats import norm, kurtosis
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import gc
from sklearn.model_selection import train_test_split
from numpy import save
from numpy import load
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC


import random as rn
import random
import math
from math import e
import os


import cv2
import PIL
from PIL import Image 
import argparse
import random as rng
import matplotlib.pyplot as plt


from keras import models
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras.applications import VGG16, VGG19, ResNet50
from keras.applications import  InceptionV3, InceptionResNetV2
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Input
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import DenseNet201, DenseNet169, Xception
from sklearn.model_selection import KFold
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras import backend as K

from Functions import*

from keras.applications import ResNet50, DenseNet201, Xception, VGG16, InceptionV3, InceptionResNetV2, MobileNetV2, NASNetMobile
import efficientnet.keras as efn 
#base_model = efn.EfficientNetB5(weights='imagenet',   pooling = max, include_top = False, input_shape = (size, size,3) )







def Resize(B_IMAGES, s_ize):
        B_IMAGES_Resized = []
        for b_image in B_IMAGES:
            b_im = cv2.resize(b_image,(s_ize, s_ize)) # Image.resize()
            B_IMAGES_Resized.append(b_im)
        return B_IMAGES_Resized



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




def Boxx(landmark, in_image):
                    left_eye = landmark
                    box = []
                    left_eye  = np.array(left_eye)
                    minn0, maxx0 = np.min(left_eye[:,0]), np.max(left_eye[:,0])
                    minn0, maxx0 = left_eye[np.argwhere(left_eye[:,0]==minn0)][0][0], left_eye[np.argwhere(left_eye[:,0]==maxx0)][0][0]
                    box.append(minn0)
                    box.append(maxx0)
                    left_eye  = np.array(left_eye)
                    minn0, maxx0 = np.min(left_eye[:,1]), np.max(left_eye[:,1])
                    minn0, maxx0 = left_eye[np.argwhere(left_eye[:,1]==minn0)][0][0], left_eye[np.argwhere(left_eye[:,1]==maxx0)][0][0]
                    box.append(minn0)
                    box.append(maxx0)
                    box = np.array(box)
                    a1, a2, a3, a4 = box[0,0], box[1,0], box[2,1], box[3,1]
                    if a1<0:a1=0
                    if a2<0:a2=0
                    if a3<0:a3=0
                    if a4<0:a4=0
                                
                    if a3>15 and a4+15<in_image.shape[0] and a1>25 and a2+25<in_image.shape[0]:
                        BOX1 = in_image[a3-8: a4+8, a1-15: a2+15,  :]
                    else: 
                        BOX1 = in_image[a3: a4, a1: a2,  :]
                    return BOX1





def Feature_Extractor (B_IMAGES, F_NAME):


                    size = len(B_IMAGES[0][0]) 
                    if size <32:
                            B_IMAGES = Resize(B_IMAGES, 32)
                            size = len(B_IMAGES[0][0]) 


                    B_IMAGES = np.array(B_IMAGES)
                    B_IMAGES = preprocess_input(B_IMAGES)
                            

                    if 1 == 2:
                        #####___ GAP___,
                        inp = Input((size,size,3))
                        backbone = ResNet50(input_tensor = inp, include_top = False)
                        x = backbone.output
                        x = GlobalAveragePooling2D()(x)#MaxPooling2D()(x)#
                        x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
                        out = Lambda(lambda x: x[:,:,0])(x)
                        model = Model(inp,out)
                        FEATURES = model.predict(B_IMAGES)
                        save(F_NAME, FEATURES)

                    if 1 == 2:
                        #####___MP___,
                        inp = Input((size,size,3))
                        backbone = VGG16(input_tensor = inp, include_top = False)
                        x = backbone.output
                        x = MaxPooling2D()(x)#
                        
                        x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
                        out = Lambda(lambda x: x[:,:,0])(x)
                        model = Model(inp,out)
                        FEATURES = model.predict(B_IMAGES)
                        FEATURES = FEATURES.reshape (FEATURES.shape[0],  FEATURES.shape[2], FEATURES.shape[1], 1)

                        a = FEATURES[:,:,0] 

                        FEATURES_ALL = np.append (FEATURES[:,:,0], FEATURES[:,:,1], axis = 1)
                        FEATURES_ALL = np.append (FEATURES_ALL, FEATURES[:,:,2], axis = 1)
                        FEATURES = FEATURES_ALL.reshape(FEATURES_ALL.shape[0],  FEATURES_ALL.shape[1])
                        save(F_NAME, FEATURES)

#  ResNet50, DenseNet201, Xception, VGG16, InceptionV3, InceptionResNetV2, MobileNetV2
#  efn.EfficientNetB5
                    if 1 == 1:
                        base_model = ResNet50(weights='imagenet',   pooling = max, include_top = False, input_shape = (size, size,3) )
                        input = Input(shape=(size, size,3), name = 'image_input')
                        x = base_model(input)
                        x = Flatten()(x)
                        model = Model(inputs=input, outputs=x)
                        FEATURES = model.predict(B_IMAGES)
                    if 1 == 1: # patches via features# PARALEL PCA 
                        b = int (len(FEATURES[0])/4)
                        Feature_Patches = [FEATURES[:,:b], FEATURES[:,b:2*b], FEATURES[:,2*b:3*b], FEATURES[:,3*b:]] 
                        Total_Features = []
                        for F_Patch in Feature_Patches:
                            # # # #---------------------------> Standarize
                            F_Patch = Standarize (F_Patch)
                            if len(F_Patch[0]) >= 2800:
                                # #____PCA____ 
                                pca = PCA(n_components=2800)
                                F_Patch = pca.fit(F_Patch).transform(F_Patch)
                            Total_Features.append(F_Patch)
                        T0, T1, T2, T3 = Total_Features[0], Total_Features[1], Total_Features[2], Total_Features[3]
                        Total_Features = np.append(T0, T1, axis = 1)
                        Total_Features = np.append(Total_Features, T2, axis = 1)
                        Total_Features = np.append(Total_Features, T3, axis = 1)
                        Total_Features = np.array(Total_Features)
                        # #____PCA____ 
                        if len(Total_Features[0]) >= 2800:
                            pca = PCA(n_components=2800)
                            Total_Features = pca.fit(Total_Features).transform(Total_Features)
                            save(F_NAME, Total_Features) #______ Save Features
                        else:
                            save(F_NAME, Total_Features) #______ Save Features  

                    if 1 == 2: # At Once
                        # # # #---------------------------> Standarize
                        if len(FEATURES[0]) >= 2800: # IN All almost cases len of features (it is 100000++) >> M, M = 2800 in this dataset 
                            FEATURES = Standarize (FEATURES)
                            # #____PCA____ 
                            pca = PCA(n_components=2800)
                            FEATURES_ = pca.fit(FEATURES).transform(FEATURES)
                            save(F_NAME, FEATURES_) #______ Save Features
                        else:
                            save(F_NAME, FEATURES) #______ Save Features





def Kernel_Tr(kernel, initial_image):
    b_im = initial_image
    b_im_inv = np.copy(b_im)
    if kernel=="Inv":
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
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,0]/((0.03)*b_im[:,:,1]), b_im[:,:,1]/((0.03)*b_im[:,:,2]), b_im[:,:,2]/((0.03)*b_im[:,:,0])
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
    elif kernel=="Sqrt":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = b_im[:,:,0]**2, b_im[:,:,1]**2, b_im[:,:,2]**2
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
    elif kernel=="exe":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = np.exp(0.05*b_im[:,:,0])*np.exp(0.05*b_im[:,:,1]), np.exp(0.03*b_im[:,:,1])*np.exp(0.03*b_im[:,:,2]), np.exp(0.01*b_im[:,:,2])*np.exp(0.01*b_im[:,:,0])
        return  b_im_inv
    elif kernel=="exe2":
        b_im_inv[:,:,0],b_im_inv[:,:,1],b_im_inv[:,:,2] = np.exp(0.01*b_im[:,:,0])*np.exp(0.02*b_im[:,:,1]), np.exp(0.04*b_im[:,:,1])*np.exp(0.01*b_im[:,:,2]), np.exp(0.01*b_im[:,:,2])*np.exp(0.01*b_im[:,:,0])
        return  b_im_inv




def Kernel_Transformation(kernel, B_IMAGES_Resized):
            B_IMAGES_Tr = []
            for init_image in B_IMAGES_Resized:
                    # plt.figure()
                    # plt.imshow(init_image)
                    # plt.show()
                    tr_image = Kernel_Tr(kernel, init_image)
                    B_IMAGES_Tr.append(tr_image)
                    # plt.figure()
                    # plt.imshow(tr_image)
                    # plt.show()
            return B_IMAGES_Tr


if 1 == 2:
        Benign = os.listdir("Dataset/Benign")
        Malignant = os.listdir("Dataset/Malignant")

        IMAGES_SKIN = []
        LABELS_SKIN = []
        for im_id in Benign:
            tr_image = np.array (Image.open("Dataset/Benign/"+im_id).convert('RGB'))
            # plt.figure()
            # plt.imshow(tr_image)
            # plt.show()
            IMAGES_SKIN.append(tr_image)
            LABELS_SKIN.append(0)
        for im_id in Malignant:
            tr_image = np.array (Image.open("Dataset/Malignant/"+im_id).convert('RGB'))
            # plt.figure()
            # plt.imshow(tr_image)
            # plt.show()
            IMAGES_SKIN.append(tr_image)
            LABELS_SKIN.append(1)
        IMAGES_SKIN = np.array(IMAGES_SKIN)
        LABELS_SKIN = np.array(LABELS_SKIN)
        ######save("IMAGES_SKIN", IMAGES_SKIN)
        ######save("LABELS_SKIN", LABELS_SKIN)


## LOAD DATA
IMAGES = load("IMAGES_SKIN.npy")
LABELS = load("LABELS_SKIN.npy").astype (int)




####### s = np.arange(LABELS.shape[0]) 
####### np.random.shuffle(s)
####### LABELS = LABELS[s]
####### IMAGES = IMAGES[s]
####### save("IMAGES_SKIN", IMAGES)
####### save("LABELS_SKIN", LABELS)



# Feature_Extractor(IMAGES, "V_MP") #Feature_Extractor(IMAGES, "V_MP")
# FEATURES = load("V_MP.npy")


# Initial_____________________________________
#Feature_Extractor (IMAGES, "_Initial_R")
#Feature_Extractor (IMAGES, "_Initial_V")

# RESIZE-BASED
if 1 ==2:
# 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350
    for _i_i_ in [400]:
        IMAGES_Resized = Resize(IMAGES, _i_i_)
        Feature_Extractor (IMAGES_Resized, "_"+str(_i_i_)+"_R")


# KERNEL-TRANSFORMATION-BASED
if 1 == 2: 
    Sizes = [100, 150, 200, 250, 300]   # 224 too
    for size in Sizes:
        kernel = "Square_Normalized"  #Square_Normalized
        IMAGES_Resized = Resize(IMAGES, size)
        IMAGES_Tranformed = Kernel_Transformation(kernel, IMAGES_Resized)
        Feature_Extractor (IMAGES_Tranformed, "_"+str(size)+str(kernel)+"_V")




if 1 == 2:
        # __________________ResNet ____________________________________
        if 1 == 1:

                _150_R = load ("ResNet_Views/Initial/_150_R.npy")
                _300_R = load ("ResNet_Views/Initial/_300_R.npy")

                _200Square_Normalized_R = load ("ResNet_Views/Square_Normalized/_200Square_Normalized_R.npy")

                _100dia_x_R = load ("ResNet_Views/dia_x/_100dia_x_R.npy")
                _200dia_x_R = load ("ResNet_Views/dia_x/_200dia_x_R.npy")
                _250dia_x_R = load ("ResNet_Views/dia_x/_250dia_x_R.npy")

                _250exe2_R = load ("ResNet_Views/x5x4x3_R/_250exe2_R.npy")

                _100_try_dia_R10__R = load ("ResNet_Views/try_dia_R10/_100_try_dia_R10__R.npy") 
          
        if 1 == 2:
        # __________________Initial _______
            FEATURES = load('_Initial_R.npy')
        # __________________ MV ____________________________________________
        if 1 == 1:
            FEATURES = np.append (_300_R, _200Square_Normalized_R, axis=1) # 91.9
            FEATURES = np.append (FEATURES, _200dia_x_R, axis=1)       # 92.2
            FEATURES = np.append (FEATURES, _250dia_x_R, axis=1)       # 92.3
            FEATURES = np.append (FEATURES, _100dia_x_R, axis=1)       # 92.4
            FEATURES = np.append (FEATURES, _250exe2_R, axis=1)      # 92.5
            FEATURES = np.append (FEATURES, _100_try_dia_R10__R, axis=1)      # 92.7



#___________________ EVALUATION ________________________________________________________________
if 1 == 1:

        RS = 0
        FINAL_ACC = []
        FINAL_F1 = []
        FINAL_ROC = []
        CONFUSION_MATRIX = np.zeros ((1,2,2))
        FINAL_CM =  np.zeros ((2,2))
        RUN = -1
        for simulation in range (1):  # Χ x ten shuffle cross validation
                        RUN += 1          

                        DATASET   = np.append(FEATURES, np.array(LABELS).reshape(-1,1), axis=1)

                        X = DATASET [:,:len(DATASET[0])-1]
                        Y = DATASET [:,len(DATASET[0])-1]
                        c1,c2 = Count_Classes2(Y)



                        ### s = np.arange(Y.shape[0]) 
                        ### np.random.shuffle(s)
                        #### X = X[s]
                        ### Y = Y[s]
                        RS += 10
                        kf = KFold(n_splits=10, shuffle=False, random_state = 0)    # <____Shuffled___10 Cross Validation
                        kf.get_n_splits(X)

                        Ts_score = []
                        Ts_score1 = []
                        Ts_score2 = []
                        Ts_score3 = []
                        Ts_F1_score = []
                        sensitivity_score = []
                        specificity_score = []
                        PPV_score = []
                        NPV_score = []
                        ROC_AUC_score = []
                        log_l_score = []

                        Confusion_Matrix = np.zeros ((10,2,2))
                        run = -1
                        for train_index, test_index in kf.split(X):
                                                    run += 1

                                                    x_train_ALL1, x_test_ALL1 = X[train_index], X[test_index]
                                                    y_train, y_test = Y[train_index], Y[test_index]


                                                    ### LR_ALL1 = SVC().fit(x_train_ALL1, y_train)
                                                    LR_ALL1 = LogisticRegression(C=1,  max_iter=250, random_state=100000).fit(x_train_ALL1, y_train) # 
                                                    pr_ts1 =  LR_ALL1.predict (x_test_ALL1)
                                                    pred1 =  LR_ALL1.predict_proba (x_test_ALL1)[:,1]

                                                    ts1, ts1, ts_F1, ts_F1 = Classification_report(pr_ts1, pr_ts1, y_test, y_test)
                                                    Acc  = accuracy_score(y_test, pr_ts1)
                                                    F1 = f1_score(y_test, pr_ts1)#
                                                    ROC_AUC = metrics.roc_auc_score(y_test, pred1) # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
                                                    CM = confusion_matrix(y_test, pr_ts1)

                                                    Confusion_Matrix[run,0,0], Confusion_Matrix[run,0,1] = CM[0,0], CM[0,1] 
                                                    Confusion_Matrix[run,1,0], Confusion_Matrix[run,1,1] = CM[1,0], CM[1,1]     

                                                    Ts_score.append      (Acc)
                                                    Ts_F1_score.append   (F1)
                                                    ROC_AUC_score.append (ROC_AUC)

                        Acc = np.round (np.mean (np.array(Ts_score)), 3)
                        F1  = np.round (np.mean (np.array(Ts_F1_score)), 3)
                        ROC_AUC  = np.round (np.mean (np.array(ROC_AUC_score)), 3)
                        CONFUSION_MATRIX[RUN,0,0], CONFUSION_MATRIX[RUN,0,1] = np.round (np.mean (Confusion_Matrix[:,0,0]), 3) , np.round (np.mean (Confusion_Matrix[:,0,1] ), 3)
                        CONFUSION_MATRIX[RUN,1,0], CONFUSION_MATRIX[RUN,1,1] = np.round (np.mean (Confusion_Matrix[:,1,0]), 3) , np.round (np.mean (Confusion_Matrix[:,1,1] ), 3) 

                        # print(Acc)
                        # print(F1)
                        # print(ROC_AUC)

                        FINAL_ACC.append (Acc)
                        FINAL_F1.append (F1)
                        FINAL_ROC.append (ROC_AUC)  

        FINAL_CM[0,0], FINAL_CM[0,1] = np.round (np.mean (CONFUSION_MATRIX[:,0,0]), 3) , np.round (np.mean (CONFUSION_MATRIX[:,0,1] ), 3)
        FINAL_CM[1,0], FINAL_CM[1,1] = np.round (np.mean (CONFUSION_MATRIX[:,1,0]), 3) , np.round (np.mean (CONFUSION_MATRIX[:,1,1] ), 3) 




        Acc = np.round (np.mean (np.array(FINAL_ACC)), 3)
        F1  = np.round (np.mean (np.array(FINAL_F1)), 3)
        ROC_AUC  = np.round (np.mean (np.array(FINAL_ROC)), 3)

        tp, tn = FINAL_CM[0,0], FINAL_CM[1,1]
        GM = (tp*tn)**(0.5)
        
        print(Acc)
        print(GM)
        print(ROC_AUC)

        # save ("CM__DFDC__Reg_Tr_Res",FINAL_CM)
        # save ("ACC__DFDC__Reg_Tr_Res",np.array(Acc))
        # save ("F1__DFDC__Reg_Tr_Res",np.array(F1))
        # save ("ROC__AUC__DFDC__Reg_Tr_Res",np.array(ROC_AUC))


        #Acc2 = (FINAL_CM[0,0] + FINAL_CM[1,1]) / (FINAL_CM[0,0] + FINAL_CM[1,1] + FINAL_CM[0,1] + FINAL_CM[1,0])

        s = 1

