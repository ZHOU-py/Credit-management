#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:22:03 2020

@author: nelson
"""
import Data_Numeric
import pandas as pd
import DataDeal
import numpy as np
from sklearn.model_selection import train_test_split
import FSVM
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import LS_FSVM
from sklearn import preprocessing
from sklearn import svm

def Attribut_rank(model):
    data = pd.read_csv('german_credit.csv')
    #    print(data.describe())
    X = data.drop(['default'],axis = 1)
    lable = data['default']
    
    df = Data_Numeric.Data_numerique(X)
    data = DataDeal.get_data(df,lable)
    Train_data,test = train_test_split(data, test_size=0.2,random_state=42)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
    
    if model=='LSFSVM':
        kernel_dict = {'type': 'RBF','sigma':0.717}
        fuzzyvalue = {'type':'Cen','function':'Lin'}
        
        #clf = FSVM.FSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
        clf = LS_FSVM.LSFSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
        m = clf._mvalue(x_train, y_train)
    elif model=='FSVM':
        kernel_dict = {'type': 'RBF','sigma':0.717}
        fuzzyvalue = {'type':'Cen','function':'Lin'}
        
        clf = FSVM.FSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
        #clf = LS_FSVM.LSFSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
        m = clf._mvalue(x_train, y_train)
    elif model=='SVM':
        clf = svm.SVC()
        
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    auc_complete = roc_auc_score(y_test, y_pred)
    
    
    #print(X.columns)
    AUC = []
    for col in X.columns:
    #Only delete one attribut
    #    X_r = X.drop([col],axis=1)
    #    df = Data_Numeric.Data_numerique(X_r)
    #    data = DataDeal.get_data(df,lable)
    #    print(df.columns)
        
    #Use only one attribut
        X_r = pd.DataFrame(X[col])
        lable[lable == 0] = -1
        df = Data_Numeric.Data_numerique(X_r)
        X_r = np.array(df)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_r = min_max_scaler.fit_transform(X_r)
        data = np.append(X_r,lable[:,None],axis=1)
    #    print(df.columns)
    #    
        
        Train_data,test = train_test_split(data, test_size=0.2,random_state=42)
        
        x_test = test[:,:-1]
        y_test = test[:,-1]
        x_train = Train_data[:,:-1]
        y_train = Train_data[:,-1]
       
        if model=='LSFSVM':
            kernel_dict = {'type': 'RBF','sigma':0.717}
            fuzzyvalue = {'type':'Cen','function':'Lin'}
            clf = LS_FSVM.LSFSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
            m = clf._mvalue(x_train, y_train)
            
        elif model=='FSVM':
            kernel_dict = {'type': 'RBF','sigma':0.717}
            fuzzyvalue = {'type':'Cen','function':'Lin'}            
            clf = FSVM.FSVM(10,kernel_dict, fuzzyvalue,'origine',4/5)
            m = clf._mvalue(x_train, y_train)
            
        elif model=='SVM':
            clf = svm.SVC()
       
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        
        auc = roc_auc_score(y_test, y_pred)
        AUC.append(auc)
     #   print(col , ':', auc)
    
    
    indices = np.argsort(AUC)[::-1]
    featurerank=[]
    for f in range(len(indices)):
        featurerank.append(X.columns[indices[f]])
    
    print('AUC complete',auc_complete)  
      
    plt.figure(figsize=(10,8))
    feature_imp = pd.Series(AUC,index=X.columns).sort_values(ascending=False)
    sns.barplot(x= feature_imp,y=feature_imp.index)
    #plt.vlines(auc_complete,feature_imp.index[19], feature_imp.index[0])
    #plt.xlim((0.65, 0.8))
    plt.xlim((0.4, 0.7))
    plt.xlabel('Feature Importance Score_AUC')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features for SVM")
    plt.legend()
    plt.show()


if __name__ == '__main__':
#    Attribut_rank('FSVM')
#    Attribut_rank('LSFSVM')
    Attribut_rank('SVM')








