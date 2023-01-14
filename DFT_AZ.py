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

from Functions import*



# Durall et al:
# Exposing Deepfakes using simble features_____________________
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
# Exposing Deepfakes using simble features_____________________



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




## LOAD DATA
IMAGES = load("IMAGES_SKIN.npy")
LABELS = load("LABELS_SKIN.npy").astype (int)



if 1 == 2:
        DFT_Azimouthial = []
        for b_im in IMAGES:
            b_im_inv = np.copy (b_im)
            a0, a1, a2 = b_im[:,:,0], b_im[:,:,1], b_im[:,:,2]
            f0, f1, f2 = np.fft.fft2(a0), np.fft.fft2(a1), np.fft.fft2(a2)
            fshift0, fshift1, fshift2 = np.fft.fftshift(f0), np.fft.fftshift(f1), np.fft.fftshift(f2)

            magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2 = 20*np.log(np.abs(fshift0)), 20*np.log(np.abs(fshift1)), 20*np.log(np.abs(fshift2))

            b_im_inv[:,:,0], b_im_inv[:,:,1], b_im_inv[:,:,2] = magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2

            a0, a1, a2 = b_im_inv[:,:,0], b_im_inv[:,:,1], b_im_inv[:,:,2]

            mean = []
            for __im in [magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2]:
                m = list(azimuthalAverage(__im, center=None))
                mean = m + mean
            DFT_Azimouthial.append(mean)
            
        DFT_Azimouthial = np.array(DFT_Azimouthial)
        DFT_Azimouthial = Standarize (DFT_Azimouthial)
        save("DFT_Azimouthial", DFT_Azimouthial)











RS = 0
FINAL_ACC = []
FINAL_F1 = []
FINAL_ROC = []
CONFUSION_MATRIX = np.zeros ((1,2,2))
FINAL_CM =  np.zeros ((2,2))
RUN = -1
for simulation in range (1):  # 10 x ten shuffle cross validation
                RUN += 1          


                FEATURES = load ("DFT_Azimouthial.npy")
 

                DATASET   = np.append(FEATURES, np.array(LABELS).reshape(-1,1), axis=1)

                X = DATASET [:,:len(DATASET[0])-1]
                Y = DATASET [:,len(DATASET[0])-1]
                c1,c2 = Count_Classes2(Y)



                s = np.arange(Y.shape[0]) 
                np.random.shuffle(s)
                X = X[s]
                Y = Y[s]
                RS += 10
                kf = KFold(n_splits=10, shuffle=True, random_state = RS)    # <____Shuffled___10 Cross Validation
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


                                            LR_ALL1 = SVC(probability=True).fit(x_train_ALL1, y_train)
                                            #LR_ALL1 = LogisticRegression(C=5,  max_iter=1000, random_state=100000).fit(x_train_ALL1, y_train) # 
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

                print(Acc)
                print(F1)
                print(ROC_AUC)

                FINAL_ACC.append (Acc)
                FINAL_F1.append (F1)
                FINAL_ROC.append (ROC_AUC)  

FINAL_CM[0,0], FINAL_CM[0,1] = np.round (np.mean (CONFUSION_MATRIX[:,0,0]), 3) , np.round (np.mean (CONFUSION_MATRIX[:,0,1] ), 3)
FINAL_CM[1,0], FINAL_CM[1,1] = np.round (np.mean (CONFUSION_MATRIX[:,1,0]), 3) , np.round (np.mean (CONFUSION_MATRIX[:,1,1] ), 3) 


tp, tn = FINAL_CM[0,0], FINAL_CM[1,1]
GM = (tp*tn)**(0.5)

Acc = np.round (np.mean (np.array(FINAL_ACC)), 3)
F1  = np.round (np.mean (np.array(FINAL_F1)), 3)
ROC_AUC  = np.round (np.mean (np.array(FINAL_ROC)), 3)
print(Acc)
print(GM)
print(ROC_AUC)

# save ("CM__DFDC__Reg_Tr_Res",FINAL_CM)
# save ("ACC__DFDC__Reg_Tr_Res",np.array(Acc))
# save ("F1__DFDC__Reg_Tr_Res",np.array(F1))
# save ("ROC__AUC__DFDC__Reg_Tr_Res",np.array(ROC_AUC))


#Acc2 = (FINAL_CM[0,0] + FINAL_CM[1,1]) / (FINAL_CM[0,0] + FINAL_CM[1,1] + FINAL_CM[0,1] + FINAL_CM[1,0])

s = 1



