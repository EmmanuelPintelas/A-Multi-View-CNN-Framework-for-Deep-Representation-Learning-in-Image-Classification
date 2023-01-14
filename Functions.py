

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-Libraries_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
 
from keras import models
from keras.models import Sequential
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from   sklearn                   import tree
from sklearn.tree import export_text
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
import pandas as pd
import random as rn
import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras import layers
from keras import optimizers
import math
from math import e

from sklearn.linear_model import LogisticRegression



import numpy as np
from sklearn import tree


def Feature_Importances_Reduction (Classifier, X, Y , Num_of_Important_Features , Feat_Reduction):

    if Classifier == 'DT':

            Classifier_FI = tree.DecisionTreeClassifier(criterion="gini", max_depth = int (Num_of_Important_Features/2), random_state = 0)
            Classifier_FI.fit(X, Y)

            Feature_importances = Classifier_FI.feature_importances_
            return Feature_Importances_F (Feature_importances, Classifier_FI, X, Y , Num_of_Important_Features , Feat_Reduction)

    if Classifier == 'LR1':

            Classifier_FI = LogisticRegression(C=1, random_state=0).fit(X, Y)
            Coef, Imp_features, Odds_Ratio = LR_Global_Explanation (Classifier_FI, 0.2)
            Feature_importances_Index = Imp_features[:,1]
            Red_X = np.zeros((len(X), len(Feature_importances_Index)))
            _index = -1
            for index in Feature_importances_Index:
                index = int (index)
                _index+=1
                Red_X[:,_index] = X[:,index]
            return Feature_importances_Index, Red_X


    if Classifier == 'LR2':

            Classifier_FI = LogisticRegression(C=1, random_state=0).fit(X, Y)
            Coef, Imp_features, Odds_Ratio = LR_Global_Explanation (Classifier_FI, 0.5)
            Feature_importances_Index = Imp_features[:,1]
            Red_X = np.zeros((len(X), len(Feature_importances_Index)))
            _index = -1
            for index in Feature_importances_Index:
                index = int (index)
                _index+=1
                Red_X[:,_index] = X[:,index]
            return Feature_importances_Index, Red_X


    if Classifier == 'LR3':

            Classifier_FI = LogisticRegression(C=1, random_state=0).fit(X, Y)
            Coef, Imp_features, Odds_Ratio = LR_Global_Explanation (Classifier_FI, 0.6)
            Feature_importances_Index = Imp_features[:,1]
            Red_X = np.zeros((len(X), len(Feature_importances_Index)))
            _index = -1
            for index in Feature_importances_Index:
                index = int (index)
                _index+=1
                Red_X[:,_index] = X[:,index]
            return Feature_importances_Index, Red_X

def Feature_Importances_F (Feature_importances, Classifier, X, Y , Num_of_Important_Features , Feat_Reduction):


            Feature_importances_List = []
            for k in  range ( len(Feature_importances) ):
                Feature_importances_List.append( [Feature_importances[k], k] ) # κρατάμε τον αριθμό του feature για να μη το χάσουμε
                # μετά τη μετατροπή της λίστας σε φθίνουσα ή αύξουσα σειρά

            Sort (Feature_importances_List, 'decreasing')

            Num_Features = []
            for n in range(Num_of_Important_Features): # Επιλογή των Ν πιο σημαντικών features
                Num_Features.append ( Feature_importances_List[n][-1] ) # Τα πρώτα Ν δλδ αφού τα βάλαμε σε φθίνουσα σειρά

            Sort (Num_Features, 'increasing')

            if Feat_Reduction == True:
                X_Reduction = [] # Δημιουργία του πίνακα χαραχτηριστικών ο οποίος θα έχει τα instances με τα N ποιο σημαντικά features
                for i in range (len(X)) :
                    X_Reduction_tmp = [X[i][Num_Features[0]]] + []
                    for j in range(len(Num_Features)-1):
                        X_Reduction_tmp =  X_Reduction_tmp + [X[i][Num_Features[j+1]] ]
                    X_Reduction .append (X_Reduction_tmp)

                return Feature_importances_List, np.array ( X_Reduction )
            else :
                return Feature_importances_List, np.array ( X )

def Sort (Element, type):
                    if type == 'increasing':
                        for j in range ( len(Element)-1 ):
                            for i in range ( len(Element)-1 ):
                                if Element[i+1] < Element[i]:
                                    tmp = Element[i]
                                    Element[i] = Element[i+1]
                                    Element[i+1] = tmp

                    if  type == 'decreasing':
                        for j in range ( len(Element)-1 ):
                            for i in range ( len(Element)-1 ):
                                if Element[i+1] > Element[i]:
                                    tmp = Element[i]
                                    Element[i] = Element[i+1]
                                    Element[i+1] = tmp




# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-Functions_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Proposed LR Explanation Framework

def LR_Global_Explanation (LR, IF):
    Coef = LR.coef_[0]
    i = -1
    Features_index = []
    for co in Coef:
        i+=1
        if abs (co) > IF:
            Features_index.append (i)

    Odds_Ratio = []
    Global_Im_Features = []
    for j in range (len(Features_index)):
        odds_ratio = e ** Coef[Features_index[j]]
        Odds_Ratio. append (odds_ratio)
        Global_Im_Features.append ([Coef[Features_index[j]], Features_index[j]])
    Global_Im_Features = np.array(Global_Im_Features)

    return Coef, Global_Im_Features, Odds_Ratio



# Καλησπέρα,
# με βάση τις τελευταίες εξελίξεις καταλήξαμε σε binary classification για πρόβλεψη επικίνδυνου καρκίνου στον εγκέφαλο, 
# (αλλιώς Glioma detection μιας και καρκίνος αυτού  του τύπου είναι κατά κόρον κακοήθης ενώ οι άλλοι κυρίως καλοήθεις). 
# Αυτή βασικά είναι και η αιτιολόγηση του να μετατρέψουμε το υπάρχον dataset που έχει τρεις τύπου καρκίνου στον εγκέφαλο 
# σε binary πρόβλημα τύπου benign vs malignant για να αποφύγουμε και το multi classification.

# Το paper θα εστιάζει κυρίως στο explainability με βάση image analysis tools και όχι στο καθαυτό συγκεκριμένο application 
# του brain tumor prediction. Αυτό θα είναι απλά ένα application case study στο τομέα του image classification όπου θα εφαρμόζουμε το
#  προτεινόμενο explanation framework. Στόχος μας είναι να προτείνουμε μια καινούργια μεθοδολογία για explainable predictions on image 
#  classification tasks με όσο το δυνατό λιγότερη απώλεια στο classification accuracy. Οπότε δε θα υπάρχει άμεση σύγκριση με βάση τις 
#  υπόλοιπες εργασίες που απλά εστιάσανε στο καθαυτό συγκεκριμένο application του brain tumor prediction, αλλά απλά θα τους αναφέρουμε.

# Ύστερα από εξονυχιστικά πειράματα και συνδυασμούς κατέληξα ότι το mimic/ meta-learning approach  που είχα εξαρχής προτείνει δε δουλεύει 
# όπως περίμενα και θα θελα, τουλάχιστον στο συγκεκριμένο πρόβλημα οπότε αν και είναι hot και πιασαρικη μεθοδολογία αναγκάστηκα να την εγκαταλείψω.

# 2ον, το decision tree που είχαμε προτείνει και στο Grey Box για explanation των predictions, δε τα πάει αρκετά καλά για αυτό το συγκεκριμένο 
# πρόβλημα και όχι λόγο του μεγάλου αριθμού των features. Συγκεκρίμενα έφτανε γύρω στο  85% με 90% accuarcy έναντι του black box νευρωνικού που έπιανε 93% με 95%  ενώ τα black box CNN transfer learning based models των άλλων papers φτάνουν 99%+ accuracy. Αντίθετα, γραμμικά μοντέλα όπως η Logistic Regression  έποιανε εξαιρετικά stable performance results με accuracy 92% με 94% σχεδόν όσο καλά πήγαινε και το νευρωνικό.

# Γενικώς, η Logistic Regression θεωρείται White Box μοντέλο όπως κάθε γραμμικό μοντέλο. Το θέμα είναι πως δε παρέχει εκ φύσεως explanations 
# όπως το decision tree στη λογική του για παράδειγμα:

#                                                                                                         if:     x1>5 and x2<2
#                                                                                                         then: y = 1
#                                                                                                         else:  y = 0.
# Τέτοια explanations είναι αυτά που θεωρούνται και "Good Explanation" μιας και δε χρειάζεται να είναι κανείς machine learning familiar 
# αλλά και επειδή αγγίζουν την ανθρώπινη λογική αφού και οι άνθρωποι τέτοιου τύπου εξηγήσεις καταλαβαίνουν αλλά και δίνουν. Επωμένος για να 
# δοθούν καλά explanations δεν αρκεί ένα μοντέλο να θεωρείται interpretable (White model) αλλά να μπορεί να δίνει τέτοιου τύπου explanations, 
# όπως το decision tree. Για παράδειγμα ένα explanation του στυλ:
#                                                                                                         if:     x1*x2>5 and 3*x1 + 2*x2 < 2
#                                                                                                         then: y = 1
#                                                                                                         else:  y = 0.
# αν και είναι interpretable μιας και φαίνεται η συνάρτηση που το ορίζει, δεν είναι εύκολα κατανοητό αλλά είναι περίπλοκο και δυσνόητο 
# στην ανθρώπινη φύση μιας και έχουμε τη τάση να αποπλέκουμε  τις μεταβλητές και να τις αναλύουμε τη κάθε μια ξεχωριστά.

# Αναλύοντας και χρησιμοποιώντας τη μαθηματική φόρμουλα της Logistic Regression έφτιαξα μία άλλη μαθηματική φόρμουλα όπου εντοπίζει τα 
# κρίσιμα σημεία στα οποία επηρεάζεται και αλλάζει άμεσα το prediction result και έφτιαξα ένα explanation framework όπου μοιάζει άμεσα με τα
# explanations που παρέχει ένα Decision Tree ώστε να θεωρείται "Good Explanation". Οπότε το υλοποίησα αυτό σε κώδικα και εισάγοντας στο μοντέλο 
# για παράδειγμα ένα οποιοδήποτε instance για πρόβλεψη με features = [x1, x2, x3, ..... xN] αυτό εκτυπώνει αυτόματα στο τερματικό το παρακάτω 
# μήνυμα:



def LR_Local_Explanations (LR, instance):

    pr  = int(LR.predict (instance.reshape(1,-1))[0])

    Coef = LR.coef_[0]
    dj_List = []
    xLRj_List = []
    for j in range (len (instance)):
        xj = instance[j]
        Sum = 0
        for k in range (len (Coef)):
            if k != j:
                ak = Coef[k] 
                Sum = ak * instance[k] + Sum
        aj = Coef[j]        
        xLRj  = (-Sum)/aj
        xLRj_List.append (xLRj)
        dj = abs (xj - xLRj)
        dj_List.append (dj)

    dj_index = []
    dj_crit = []
    xj_crit = [] # critical features 
    dth = 5
    i = -1

    print ('')
    print ('')
    print ('         Explanation and Reasoning')
    print ('You are predicted as ' + str(pr) + " because: ")
    print ('{')
    for dj in dj_List:
        i+=1
        if abs (dj) < dth:
            dj_index.append (i)
            dj_crit.append (dj_List[i])
            xj_crit.append (xLRj_List[i])

            subrule = np.round (xLRj_List[i], 1)

            if xLRj_List[i] > instance[i]:
                    print (str (subrule) + " > " + " x" + str (i) )
            else:
                    print (str (subrule) + " < " + " x" + str (i) )  
    print ('else you will be predicted as  ' + str (abs(pr-1)))
    print ('Note: All other features remain stable for every sub rule')   
    print ('}')
    print ('Every instance that follows this rule is considered simiral to you and will be predicted as  ' + str (pr))
    print ('')


    print ('Warning!')
    feature_very_crit = []
    i=-1
    for dj in dj_List:
        i+=1
        if abs (dj) < 3:
            feature_very_crit.append(i)

    print ('The features: ')    
    for f in feature_very_crit:
            print ("x" + str (f) + "  ")  
    print ('are quite vital for you and determine in a very significant way your prediction result.')
    print ('More specifically, if your feature' + " x" + str (f) + ' changes by a value ' + str (np.round(dj_List[f],1)) + ', then your prediction result will change to '+ str (abs(pr-1)))
    print ('')
    print ('')
    print ('')
    s = 1
        


  









def Classification_report(pr_tr, pr_ts, y_train, y_test):
    predtR = pr_tr
    predtS = pr_ts
    # #print(classification_report(y_test, predtS))
    print (confusion_matrix(y_test, predtS))
    Train_Score_Trained_model = accuracy_score (y_train, predtR)
    Test_Score_Trained_model  = accuracy_score (y_test, predtS)

    # Train_F1_Score_Trained_model = f1_score (y_train, predtR)
    # Test_F1_Score_Trained_model  = f1_score (y_test, predtS)

    return (Train_Score_Trained_model, Test_Score_Trained_model, Train_Score_Trained_model, Test_Score_Trained_model)


def Create_Binary_Dataset (X, Y):

        #-------------------------------> One vs all ensemble data
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        X3 = []
        Y3 = []
        for i in range (len (X)):
                if Y[i] == 1:
                        X1.append (X[i])
                        Y1.append (Y[i])
                if Y[i] == 2:
                        X2.append (X[i])
                        Y2.append (Y[i])
                if Y[i] == 3:
                        X3.append (X[i])
                        Y3.append (Y[i])


        # X12 = np.append (X1, X2, axis = 0)
        # Y12 = np.append (Y1, Y2, axis = 0)
        X13 = np.append (X1, X3, axis = 0)
        Y13 = np.append (Y1, Y3, axis = 0)
        # X23 = np.append (X2, X3, axis = 0)
        # Y23 = np.append (Y2, Y3, axis = 0)



        X3_12 = []
        X2_13 = []
        X1_23 = []
        Y3_12 = []
        Y2_13 = []
        Y1_23 = []
        # for x3, y3 in zip (X3, Y3):
        #                 X3_12.append (x3)
        #                 Y3_12.append (y3)
        # for x12, y12 in zip (X12, Y12):
        #                 X3_12.append (x12)
        #                 Y3_12.append (0)

        for x2, y2 in zip (X2, Y2):
                        X2_13.append (x2)
                        Y2_13.append (y2)
        for x13, y13 in zip (X13, Y13):
                        X2_13.append (x13)
                        Y2_13.append (0)

        # for x1, y1 in zip (X1, Y1):
        #                 X1_23.append (x1)
        #                 Y1_23.append (y1)
        # for x23, y23 in zip (X23, Y23):
        #                 X1_23.append (x23)
        #                 Y1_23.append (0)
        #X3_12 = np.array (X3_12)
        X2_13 = np.array (X2_13)
        # X1_23 = np.array (X1_23)
        # Y3_12 = np.array (Y3_12)
        Y2_13 = np.array (Y2_13)
        # Y1_23 = np.array (Y1_23)

        #dataset3_12 = np.append (X3_12, Y3_12.reshape (-1,1), axis = 1)
        dataset2_13 = np.append (X2_13, Y2_13.reshape (-1,1), axis = 1)
        #dataset1_23 = np.append (X1_23, Y1_23.reshape (-1,1), axis = 1)

        # dataset12 = np.append (X12, Y12.reshape (-1,1), axis = 1)
        # dataset13 = np.append (X13, Y13.reshape (-1,1), axis = 1)
        # dataset23 = np.append (X23, Y23.reshape (-1,1), axis = 1)

        return dataset2_13



# # #-------------------------------> Create_Binary_Dataset 2_13 (Glioma: 1 vs No Glioma: 0)
def Convert_to_Binary_Dataset (dataset):
        X = dataset [:,:len(dataset[0])-1]
        Y = dataset [:,len(dataset[0])-1]
        dataset_2_13 = Create_Binary_Dataset (X, Y)


        X_2_13 = dataset_2_13 [:,:len(dataset_2_13[0])-1]
        Y_2_13 = dataset_2_13 [:,len(dataset_2_13[0])-1]
        for i in range (len (X)):
                if Y_2_13[i] == 2:
                        Y_2_13[i] = 1
        return X_2_13, Y_2_13



def Count_Classes3 (Y):
    cnt1, cnt2, cnt3 = 0,0,0
    for v in Y:
        if v == 1:
            cnt1 += 1
        elif v == 2:
            cnt2  += 1
        elif v == 3:
            cnt3  += 1
    return cnt1, cnt2, cnt3

def Count_Classes2 (Y):
    cnt0, cnt1= 0,0
    for v in Y:
        if v == 0:
            cnt0 += 1
        elif v == 1:
            cnt1  += 1
    print ("1: "+str(cnt1))

    return cnt0, cnt1


def PrintInFile(X, y, FileName):
    
    file = open(FileName,"w")
    for i in range(len(X)):
        for value in X[i]:
            file.write ("%10.4f,"%value)
        file.write("  %i\n" % y[i])
    file.close()

def Sort (Element, type):
                    if type == 'increasing':
                        for j in range ( len(Element)-1 ):
                            for i in range ( len(Element)-1 ):
                                if Element[i+1] < Element[i]:
                                    tmp = Element[i]
                                    Element[i] = Element[i+1]
                                    Element[i+1] = tmp

                    if  type == 'decreasing':
                        for j in range ( len(Element)-1 ):
                            for i in range ( len(Element)-1 ):
                                if Element[i+1] > Element[i]:
                                    tmp = Element[i]
                                    Element[i] = Element[i+1]
                                    Element[i+1] = tmp
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
