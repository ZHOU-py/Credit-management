#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:32:39 2020

@author: nelson
"""

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import Precision
import DataDeal
import GridSearch_parametre
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import FSVM_bagging
import LS_FSVM_bagging
import FSVM
import LS_FSVM
import BFSVM
import WLSSVM
import BLSFSVM

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

data = pd.read_csv("data/Database_Encodage.csv")
#    data = pd.read_csv("data/Database_label.csv")
#    data = pd.read_csv("data/Database_onehotencoder.csv")
X = data.drop(['Loan classification'],axis = 1)
label = data['Loan classification']
data = DataDeal.get_data(X,label,'scaler','True')

Train_data,test = train_test_split(data, test_size=0.3,random_state = 42)

x_test = test[:,:-1]
y_test = test[:,-1]
x_train = Train_data[:,:-1]
y_train = Train_data[:,-1]


print('SVM')
#clf = svm.SVC(C = 3,class_weight='balanced')
clf = svm.SVC(C = 3)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#Precision.accuracy(y_pred,y_test)
Precision.precision(y_pred,y_test)

print('*****************')

print('LinearSVM')
#clf = svm.LinearSVC(C=3,class_weight='balanced')
clf = svm.LinearSVC(C=3)
#clf = OneVsOneClassifier(clf)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
#Precision.accuracy(y_pred,y_test)
Precision.precision(y_pred,y_test)

print('*****************')

print('FSVM')
kernel_dict = {'type': 'RBF','sigma':0.717}
fuzzyvalue = {'type':'Hyp','function':'Exp'}

clf = FSVM.FSVM(3,kernel_dict, fuzzyvalue,'UpSampling',3/4)
m = clf._mvalue(x_train, y_train)

clf = OneVsOneClassifier(clf)
clf.fit(x_train, y_train)
#with open('save/LSFSVM_Cen_Lin_RBF_Origine1.pickle', 'wb') as f:
#    pickle.dump(clf, f)
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)
Precision.precision(y_pred,y_test)


print('*****************')

print('BFSVM')
kernel_dict = {'type': 'RBF','sigma':0.717}
fuzzyvalue = {'type':'Hyp','function':'Probit'}

clf = BFSVM.BFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')
m = clf._mvalue(x_train, y_train)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)

Precision.precision(y_pred,y_test)


print('*****************')

print('LSFSVM')
kernel_dict = {'type': 'RBF','sigma':0.717}
fuzzyvalue = {'type':'Hyp','function':'Exp'}

clf = LS_FSVM.LSFSVM(3,kernel_dict, fuzzyvalue,'UpSampling',3/4)
m = clf._mvalue(x_train, y_train)
clf.fit(x_train, y_train)
#with open('save/LSFSVM_Cen_Lin_RBF_Origine1.pickle', 'wb') as f:
#    pickle.dump(clf, f)
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)
Precision.precision(y_pred,y_test)


print('*****************')

print('LSSVM')

kernel_dict = {'type': 'RBF','sigma':0.717}
fuzzyvalue = {'type':'Hyp','function':'Exp'}

clf = WLSSVM.LSFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')

alpha,b,e = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)

Precision.precision(y_pred,y_test)

print('*****************')

print('WLSSVM')    
    
v1 = clf.weights(e)
alpha,b = clf.weightsleastSquares(v1, x_train, y_train)
 
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)

Precision.precision(y_pred,y_test)

print('*****************')

print('LSBFSVM')

kernel_dict = {'type': 'RBF','sigma':0.717}
fuzzyvalue = {'type':'Hyp','function':'Probit'}

clf = BLSFSVM.LSBFSVM(10,kernel_dict, fuzzyvalue,'UpSampling')
m = clf._mvalue(x_train, y_train)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)
decision_function = clf.decision_function(x_test)
Precision.precision(y_pred,y_test)

print('*****************')
    
print('SVM_bagging')
clf = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf',class_weight='balanced'),\
                         n_estimators=10, random_state=0).fit(x_train, y_train)
#with open('save/SVM_bagging.pickle', 'wb') as f:
#    pickle.dump(clf, f)
    
y_pred = clf.predict(x_test)
Precision.precision(y_pred,y_test)

print('*****************')

print('FSVM_bagging')
fuzzyvalue = {'type':'Cen','function':'Lin'} 
param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,1,30)}
#    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
databalance = 'Origine'
clf = FSVM_bagging.FSVM_bagging(7,databalance,'RBF',param_grid,'Acc', fuzzyvalue, 3/4,1) 
clf.fit(x_train,y_train)

#with open('save/FSVMbag_Cen_Lin_RBF_Origine.pickle', 'wb') as f:
#    pickle.dump(clf, f)
    
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)
Precision.precision(y_pred,y_test)

print('*****************')

print('LSFSVM_bagging')
fuzzyvalue = {'type':'Cen','function':'Lin'} 
param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,1,30)}
#    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
databalance = 'Origine'
clf = LS_FSVM_bagging.LS_FSVM_bagging(7,databalance,'RBF',param_grid,'Acc', fuzzyvalue, 3/4,1) 
clf.fit(x_train,y_train)

#with open('save/LSFSVMbag_Cen_Lin_RBF_Origine.pickle', 'wb') as f:
#    pickle.dump(clf, f)
    
y_pred = clf.predict(x_test)
y_prob = clf.predict_prob(x_test)
Precision.precision(y_pred,y_test)







