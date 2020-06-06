#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:31:06 2020

@author: nelson
"""


from sklearn.ensemble import BaggingClassifier
import DataDeal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
import FSVM
import LS_FSVM
import BFSVM
import WLSSVM
import BLSFSVM
import FSVM_bagging
import LS_FSVM_bagging
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SVMSMOTE

import matplotlib.pyplot as plt
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data/Bin_onhotencoder.csv")
#data = pd.read_csv("data/Database_onehotencoder.csv")

#data = pd.read_csv("data/Onehotencoder_loan.csv")
#data = pd.read_csv("data/Onehotencoder_loan_ins0.csv")
#X = data.drop(['Loan classification'],axis = 1)

#data = pd.read_csv("data/Onehotencoder_overdraft.csv")
#data = data.drop(['Unnamed: 0'],axis = 1)

'''
q_low = data["Other installments of other loans-Converted"].quantile(0.01)
print("q_low",q_low)
q_hi = data["Other installments of other loans-Converted"].quantile(0.99)
print("q_low",q_hi)


# Overdraft
data= data[(data["Other installments of other loans-Converted"] < q_hi) & (\
                data["Other installments of other loans-Converted"] > q_low)]


# Loan
data1 = data[(data["Other installments of other loans-Converted"] < q_hi) & 
            (data["Other installments of other loans-Converted"] > q_low) &
            (data["Loan classification"] == 1)]
    
data2 = data[(data["Other installments of other loans-Converted"] <= q_low) &
             (data["Loan classification"] == 1)]
    
number = data2.index
index = np.random.randint(len(data2.index),size=int(len(data2.index)*0.02))
data2_prim = data2.loc[number[index].values]
    
data3 = data[(data["Other installments of other loans-Converted"] < q_hi) &\
             (data["Other installments of other loans-Converted"] > q_low)&\
             (data["Loan classification"] == -1)]
    
data4 = data[(data["Other installments of other loans-Converted"] >= q_hi) & ( data["Loan classification"] == -1)]

data5 = data[(data["Other installments of other loans-Converted"] <= q_low) & (data["Loan classification"] == -1)]
pdList = [data1, data2_prim, data3, data4, data5]  # List of your dataframes
data = pd.concat(pdList)



# Seperate Other installment en couches
install = data["Other installments of other loans-Converted"][data["Other installments of other loans-Converted"]!=0]
installment = data["Other installments of other loans-Converted"]
#plt.boxplot(install)
#plt.boxplot(installment)



data["Other installments of other loans-Converted"][(data["Other installments of other loans-Converted"]<10000) &\
                                                    (data["Other installments of other loans-Converted"]!=0)]= 1
data["Other installments of other loans-Converted"][(data["Other installments of other loans-Converted"]>=10000) &\
                                                    (data["Other installments of other loans-Converted"]!=0)]= 2

    
X = data.drop(['Loan classification','Approved loan','Duration (in months)',
               'Monthly Installment of the loan-Converted',
               'Other installments of other loans-Converted',
               'Monthly Income '],axis = 1)
'''
    
X = data.drop(['Loan classification'],axis = 1)
label = data['Loan classification']


###
# Not use PCA
X1 = X.astype('int64')
X1 = np.array(X1)
lable = label.values
X1 = (X1 - X1.mean())/X1.std()

data = np.append(X1,lable[:,None],axis=1)

###

#data = DataDeal.get_data(X,label,'normaliser','False')
print('Good clients %s'%Counter(label))

Train_data,test = train_test_split(data, test_size=0.3,shuffle = True)

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
    clf = LinearDiscriminantAnalysis(solver='lsqr')
    clf = clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def QDA():
    print('Q Discriminant Analysis')
    clf = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)
    
def KNN():
    print('k-nearest Neighbors')
    parameters={'n_neighbors':range(5,10,1),'weights':['uniform','distance'],'leaf_size':range(30,50,5)}
    clf = KNeighborsClassifier(n_neighbors=5,weights='distance',p=2)
    clf = GridSearchCV(clf,parameters,scoring='roc_auc',cv=5).fit(x_train, y_train)
    print(clf.best_params_)
    y_pred = clf.predict(x_test)    
    Precision.precision(y_pred,y_test)
    
def DT():
    print('Decision Tree')
    parameters={'min_samples_split':np.arange(0,1,0.1),'criterion':['entropy']}
    clf = DecisionTreeClassifier()
    clf = GridSearchCV(clf,parameters,scoring='roc_auc',cv=5).fit(x_train, y_train)
    print(clf.best_params_)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)

def MLP():
    print('Neural Network')
    x_tra = x_train.copy()
    scaler = StandardScaler() # 标准化转换
    scaler.fit(x_tra)  # 训练标准化对象
    x_tra = scaler.transform(x_tra)  
    clf = MLPClassifier(solver='lbfgs', alpha=1e-2,hidden_layer_sizes=(10, 5))
    clf = clf.fit(x_tra, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)

def RF():
    print('Random Forest')
    clf = RandomForestClassifier().fit(x_train, y_train)
 #   clf = RandomForestClassifier(max_leaf_nodes=20,\
 #   n_estimators =  171,criterion = 'entropy',min_samples_leaf=21,min_samples_split=16,\
 #   max_depth=3, max_features=5 ,oob_score = True,class_weight='balanced', random_state=42)
    clf.fit(x_train,y_train)

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
    
def GradientBoost():
    print('Gradient Boost')
    X, Y = SVMSMOTE(random_state=42).fit_sample(x_train, y_train)
#    clf = GradientBoostingClassifier(learning_rate=0.005, n_estimators=400,max_depth=11,\
#                                     min_samples_leaf =70, min_samples_split =1000, \
#                                     max_features='sqrt', subsample=1, random_state=10)
    clf = GradientBoostingClassifier()
    clf.fit(X, Y)
    ypred = clf.predict(x_test)
    Precision.precision(ypred,y_test)
    
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
    print('FSVM')
    kernel_dict = {'type': 'RBF','sigma':2}
    fuzzyvalue = {'type':'Hyp','function':'Exp'}
    
    clf = FSVM.FSVM(8,kernel_dict, fuzzyvalue,'UpSampling',1,1)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)

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
    fsvm = FSVM.FSVM(C, kernel_dict, fuzzyvalue,'UpSampling', 3/4)
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
    print('LSFSVM')
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = LS_FSVM.LSFSVM(3,kernel_dict, fuzzyvalue,'UpSampling',1)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    Precision.precision(y_pred,y_test)

print('******************')

def SVM_GS():
    print('SVM_GS')
    parameters = {'C':range(1,10,1),'gamma': np.logspace(-1, 1, 30)}
    clf = GridSearchCV(svm.SVC(C=3,class_weight='balanced'), parameters).fit(x_train, y_train)

#    print(clf.best_params_)
    #{'C': 2.9470517025518106, 'gamma': 2.329951810515372}
    
#    clf = svm.SVC(C=3,class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)

print('******************')

def LinearSVC():
    print('LinearSVC')
    clf = svm.LinearSVC(C=3,class_weight='balanced')
    parameters = {'C':range(1,10,1),'loss':['hinge','squared_hinge']}
    clf = GridSearchCV( svm.LinearSVC(class_weight='balanced'), parameters).fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    Precision.precision(y_pred,y_test)
print('******************')


def BilateralFSVM():
    print('BFSVM')
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Hyp','function':'Probit'}
    
    clf = BFSVM.BFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    Precision.precision(y_pred,y_test)
    
def WeightedLSSVM():
    print('WLSSVM')    
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = WLSSVM.LSFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')
    alpha,b,e = clf.fit(x_train, y_train)
    v1 = clf.weights(e)
    alpha,b = clf.weightsleastSquares(v1, x_train, y_train)
     
    y_pred = clf.predict(x_test)
    
    Precision.precision(y_pred,y_test)

def LSSVM():
    print('LSSVM')
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = WLSSVM.LSFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')
    
    alpha,b,e = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    
    Precision.precision(y_pred,y_test)
    
def LSBFSVM():
    print('LSBFSVM')

    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Hyp','function':'Probit'}
    
    clf = BLSFSVM.LSBFSVM(3,kernel_dict, fuzzyvalue,'UpSampling')
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)



def SVM_bagging():
    print('SVM_bagging')
    clf = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf',class_weight='balanced'),\
                             n_estimators=10, random_state=0).fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)



def FSVM_bag():
    print('FSVM_bagging')
    fuzzyvalue = {'type':'Cen','function':'Lin'} 
    param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,1,30)}
    #    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
    databalance = 'Origine'
    clf = FSVM_bagging.FSVM_bagging(7,databalance,'RBF',param_grid,'Acc', fuzzyvalue, 3/4,1) 
    clf.fit(x_train,y_train)
       
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)

def LSFSVM_bagging():
    print('LSFSVM_bagging')
    fuzzyvalue = {'type':'Cen','function':'Lin'} 
    param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,1,30)}
    #    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
    databalance = 'Origine'
    clf = LS_FSVM_bagging.LS_FSVM_bagging(7,databalance,'RBF',param_grid,'Acc', fuzzyvalue, 3/4,1) 
    clf.fit(x_train,y_train)
        
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)


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
    type2 = metrics.precision_score(y_test, ypred)
    good = metrics.recall_score(y_test, ypred)
    auc = roc_auc_score(y_test, ypred)
    cm = confusion_matrix(y_test,ypred)
    bad = cm[0,0]/(cm[0,0]+cm[0,1])
    type1 = cm[0,0]/(cm[0,0]+cm[1,0])
    print('bad precision',round(bad,3),'good precision',round(good,3),\
              'type1', round(type1,3), 'type2', round(type2,3), 'total accuracy', round(score,3))
    print('AUC:', auc)


LR()
print('******************')
LDA()
print('******************')

QDA()
print('******************')

KNN()
print('******************')

MLP()
print('******************')

DT()
print('******************')

RF()
print('******************')

AdaBoost()
print('******************')

GausNB()
print('******************')

SVM_GS()
print('******************')

LinearSVC()

print('******************')

GradientBoost()
print('******************')


FuzzySVM()
print('******************')


BilateralFSVM()
print('******************')


LSSVM()
print('******************')

LSFuzzySVM()
print('******************')

WeightedLSSVM()
print('******************')


LSBFSVM()
print('******************')


SVM_bagging()
print('******************')

FSVM_bag()
print('******************')

LSFSVM_bagging()
print('******************')






