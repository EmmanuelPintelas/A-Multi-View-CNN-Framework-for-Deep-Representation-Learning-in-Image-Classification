import efficientnet.keras as efn 
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
from Functions import*
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





resnet         =  load("Ensemble_Features/ResNet.npy")
inception      =  load("Ensemble_Features/InceptionV3.npy")
vgg           =  load("Ensemble_Features/VGG.npy")                 
efficient      =  load("Ensemble_Features/EfficientNetB5.npy")    
mobile         =  load("Ensemble_Features/MobileNetV2.npy")           
densenet        = load("Ensemble_Features/DenseNet.npy")      
xception        = load("Ensemble_Features/Xception.npy")
inc_res         = load("Ensemble_Features/InceptionResNetV2.npy")

Y = load("LABELS_SKIN.npy")



X = resnet
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=False, random_state=0)
kf.get_n_splits(X)

Acc_List = []
F1_List = []
cnt = 0
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
        _train_y, _test_y = Y[train_index], Y[test_index]

        resnet_train, resnet_test       = resnet[train_index], resnet[test_index]
        inception_train, inception_test = inception[train_index], inception[test_index]
        vgg_train, vgg_test             = vgg[train_index], vgg[test_index]
        efficient_train, efficient_test = efficient[train_index], efficient[test_index]
        mobile_train, mobile_test       = mobile[train_index], mobile[test_index]
        densenet_train, densenet_test   = densenet[train_index], densenet[test_index]
        xception_train, xception_test   = xception[train_index], xception[test_index]
        inc_res_train, inc_res_test     = inc_res[train_index], inc_res[test_index]



        
        pred_res = LogisticRegression(C=5,  max_iter=2000, random_state=100000).fit(resnet_train, _train_y).predict_proba (resnet_test)[:,1]
        pred_inc = LogisticRegression(C=5,  max_iter=2000, random_state=100000).fit(inception_train, _train_y).predict_proba (inception_test)[:,1]
        pred_vgg = LogisticRegression(C=5,  max_iter=2000, random_state=100000). fit(vgg_train, _train_y).predict_proba (vgg_test)[:,1]
        pred_eff = LogisticRegression(C=5,  max_iter=2000, random_state=100000).fit(efficient_train, _train_y).predict_proba (efficient_test)[:,1]
        pred_mob = LogisticRegression(C=5,  max_iter=2000, random_state=100000). fit(mobile_train, _train_y).predict_proba (mobile_test)[:,1]
        pred_den = LogisticRegression(C=5,  max_iter=2000, random_state=100000).fit(densenet_train, _train_y).predict_proba (densenet_test)[:,1]
        pred_xce = LogisticRegression(C=5,  max_iter=2000, random_state=100000).fit(xception_train, _train_y).predict_proba (xception_test)[:,1]
        pred_inc_res = LogisticRegression(C=5,  max_iter=2000, random_state=100000). fit(inc_res_train, _train_y).predict_proba (inc_res_test)[:,1]

        pred_F1 =   (pred_res)/1
        pred_F2 =   (pred_res + pred_den)/2
        pred_F4 =   (pred_res + pred_vgg  + pred_den + pred_eff)/4
        pred_F8 =   (pred_res + pred_vgg  + pred_inc + pred_den + pred_mob + pred_eff + pred_xce + pred_inc_res)/8

        pred_F = pred_F8
        pr_ts = np.copy(pred_F)
        pr_ts[np.argwhere(pred_F<=0.5)] = 0
        pr_ts[np.argwhere(pred_F>0.5)] = 1

        Acc = accuracy_score(_test_y, pr_ts) 
        print(np.round (Acc, 3))

        tn, fp, fn, tp = confusion_matrix(_test_y, pr_ts).ravel()
        GM = (tp*tn)**(0.5)
        sensitivity =  tp / (tp + fn) # sensitivity, recall
        specificity = tn / (tn+fp) # specificity, selectivity
        PPV = tp / (tp + fp) # precision or positive predictive value (PPV)
        NPV =  tn / (tn + fn)# negative predictive value (NPV)
        ROC_AUC = metrics.roc_auc_score(_test_y, pred_F) # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        Ts_score.append (Acc)
        sensitivity_score.append (sensitivity)
        specificity_score.append (specificity)
        PPV_score.append (PPV)
        NPV_score.append (NPV)
        ROC_AUC_score.append (ROC_AUC)
        GM_score.append (GM)

    Acc = np.round (np.mean (np.array(Ts_score)), 3)
    F1  = np.round (np.mean (np.array(Ts_F1_score)), 3)
    sensitivity = np.round (np.mean (np.array(sensitivity_score)), 3)
    specificity  = np.round (np.mean (np.array(specificity_score)), 3)
    PPV = np.round (np.mean (np.array(PPV_score)), 3)
    NPV  = np.round (np.mean (np.array(NPV_score)), 3)
    ROC_AUC  = np.round (np.mean (np.array(ROC_AUC_score)), 3)
    GM  = np.round (np.mean (np.array(GM_score)), 3)

    print(Acc)
    print(GM)
    print(ROC_AUC)
    print(sensitivity)
    print(specificity)
    print(PPV)
    print(NPV)




