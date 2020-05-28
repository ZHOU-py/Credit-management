#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:31:06 2020

@author: nelson
"""


from sklearn.ensemble import BaggingClassifier
import LS_FSVM
import FSVM
import DataDeal
from sklearn.model_selection import train_test_split
from sklearn import svm
import Precision
import xgboost as xgb
import pandas as pd
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
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("data/Database_onehotencoder.csv")

X = data.drop(['Loan classification'],axis = 1)
label = data['Loan classification']
data = DataDeal.get_data(X,label,'normaliser','False')

Train_data,test = train_test_split(data, test_size=0.2,shuffle = True)

x_test = test[:,:-1]
y_test = test[:,-1]
x_train = Train_data[:,:-1]
y_train = Train_data[:,-1]


data_maj = Train_data[y_train == 1]  # 将多数
data_min =  Train_data[y_train != 1] 
index = np.random.randint(len(data_maj), size=int(len(data_min)*2)) 
lower_data_maj = data_maj[list(index)]
Train_data = np.append(lower_data_maj,data_min,axis=0)
x_train = Train_data[:,:-1]
y_train = Train_data[:,-1]



def LR():
    print('Linear Regression')
    reg = LinearRegression().fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    y_pred = np.sign(y_pred)
    Precision.precision(y_pred,y_test)
    
def LDA():
    print('Linear Discriminant Analysis')
    clf = LinearDiscriminantAnalysis().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def QDA():
    print('Linear Discriminant Analysis')
    clf = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def KNN():
    print('k-nearest Neighbors')
    clf = KNeighborsClassifier(weights='distance',p=1).fit(x_train, y_train)
    y_pred = clf.predict(x_test)    
    Precision.precision(y_pred,y_test)
    
def DT():
    print('Decision Tree')
    clf = DecisionTreeClassifier().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)

def MLP():
    print('Neural Network')
    clf = MLPClassifier(alpha=1, max_iter=100).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)

def RF():
    print('Random Forest')
    clf = RandomForestClassifier().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)

def AdaBoost():
    print('Ada Boost')
    clf = AdaBoostClassifier().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def GausNB():
    print('Gaussian Naive Bayes')
    clf = GaussianNB().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def LSFSVM_GS():
    print('LSFSVM_GS')
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    param_grid = {'C': range(1,10,1), 'sigma': np.logspace(-1, 1, 30)}
    
    C,sigma = GridSearch_parametre.LS_FSVM_best(x_train,y_train,'RBF',param_grid,'AUC',fuzzyvalue, 3/4, 1)
    kernel_dict = {'type': 'RBF','sigma':sigma} 
    
    lssvm = LS_FSVM.LSFSVM(C,kernel_dict,fuzzyvalue,'origine',3/4)
    #lssvm = LS_FSVM.LSFSVM(10, kernel_dict, fuzzyvalue, 3/4)
    lssvm._mvalue(x_train, y_train)
    lssvm.fit(x_train, y_train)
    
    #with open('save/LSFSVM_Cen_Lin_RBF_Origine.pickle', 'wb') as f:
    #    pickle.dump(lssvm, f)
    
    y_pred = lssvm.predict(x_test)
    y_prob = lssvm.predict_prob(x_test)
    print(y_prob[:10])
    Precision.precision(y_pred,y_test)
    
print('******************')

def FuzzySVM():
    print('LSFSVM')
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = FSVM.FSVM(10,kernel_dict, fuzzyvalue,'o',1)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    #with open('save/LSFSVM_Cen_Lin_RBF_Origine1.pickle', 'wb') as f:
    #    pickle.dump(clf, f)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    Precision.precision(y_pred,y_test)
    
print('******************')


def FSVM_GS():

    print('FSVM_GS')
    #kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    param_grid = {'C': range(1,10,1), 'sigma': np.logspace(-1, 1, 30)}
    C,sigma = GridSearch_parametre.LS_FSVM_best(x_train,y_train,'RBF',param_grid,'AUC',fuzzyvalue, 3/4, 1)
    kernel_dict = {'type': 'RBF','sigma':sigma}
    #fsvm = FSVM.FSVM(10, kernel_dict, fuzzyvalue, 3/4)
    fsvm = FSVM.FSVM(C, kernel_dict, fuzzyvalue,'origine', 3/4)
    m_value = fsvm._mvalue(x_train, y_train)
    fsvm.fit(x_train, y_train)
    
    #with open('save/FSVM_Cen_Lin_RBF_Origine.pickle', 'wb') as f:
    #    pickle.dump(fsvm, f)
        
        
    y_pred = fsvm.predict(x_test)
    y_prob = fsvm.predict_prob(x_test)
    print(y_prob[:10])
    Precision.precision(y_pred,y_test)
print('******************')

def LSFuzzySVM():
    print('FSVM')
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = LS_FSVM.LSFSVM(10,kernel_dict, fuzzyvalue,'o',1)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    #with open('save/LSFSVM_Cen_Lin_RBF_Origine1.pickle', 'wb') as f:
    #    pickle.dump(clf, f)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    Precision.precision(y_pred,y_test)

print('******************')

def SVM_GS():
    print('SVM_GS')
    parameters = {'C':range(1,10,1),'gamma': np.logspace(-1, 1, 30)}
    clf = GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    
    #with open('save/SVM_RBF_GS.pickle', 'wb') as f:
    #    pickle.dump(clf, f)
    print(clf.best_params_)
    #{'C': 2.9470517025518106, 'gamma': 2.329951810515372}
    
    clf = svm.SVC(probability = True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    print('y_prob',y_prob[:10])
    Precision.precision(y_pred,y_test)

print('******************')

def LinearSVC():
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    y_prob = clf.decision_function(x_test)
    #print(y_pred[y_prob<0])
    #print('y_prob',y_prob)
    Precision.precision(y_pred,y_test)
print('******************')



def SVM_bagging():
    print('SVM_bagging')
    clf = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf'),\
                             n_estimators=10, random_state=0).fit(x_train, y_train)
    #with open('save/SVM_bagging.pickle', 'wb') as f:
    #    pickle.dump(clf, f)
        
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
print('******************')

def FSVM_bag():
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
print('******************')

def LSFSVL_bagging():
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

print('******************')

def XGBOOST():
    print('XGBOOST')
    #'binary:logistic'
    #'reg:gamma'   # more balance between the binary class
    params_xgb2 = {'max_depth': 6,'eta': 0.025,'silent':1, 'objective':'reg:gamma'  ,\
                   'eval_matric': 'auc', 'minchildweight': 10.0,'maxdeltastep': 1.8,\
                   'colsample_bytree': 0.4,'subsample': 0.8,'gamma': 0.71,'numboostround' : 391}
    
    y_train[y_train==-1]=0
    y_test[y_test==-1]=0
    dtrain = xgb.DMatrix(x_train,y_train)
    #dvalid = xgb.DMatrix(Train_data,label=label)
    evals = [(dtrain, 'train')]
    xgboost = xgb.train(params_xgb2, dtrain, 10, evals)
    
    #with open('save/XGBOOST.pickle', 'wb') as f:
    #    pickle.dump(xgboost, f)
        
    dtest = xgb.DMatrix(x_test)
    ypred = xgboost.predict(dtest)
    ypred[ypred>=0.5]=1
    ypred[ypred<0.5]=0
    score = accuracy_score(y_test, ypred)
    type1 = metrics.precision_score(y_test, ypred)
    good = metrics.recall_score(y_test, ypred)
    auc = roc_auc_score(y_test, ypred)
    cm = confusion_matrix(y_test,ypred)
    bad = cm[0,0]/(cm[0,0]+cm[0,1])
    type2 = cm[0,0]/(cm[0,0]+cm[1,0])
    print('bad precision',round(bad,3),'good precision',round(good,3),\
              'type1', round(type1,3), 'type2', round(type2,3), 'total accuracy', round(score,3))
    print('AUC:', auc)





LR()
LDA()
QDA()
KNN()
DT()


MLP()
RF()

AdaBoost()

GausNB()



FuzzySVM()
FSVM_GS()


LSFuzzySVM()

SVM_GS()






