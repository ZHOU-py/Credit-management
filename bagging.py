#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:54:16 2020

@author: nelson
"""
import random
import numpy as np


import DataDeal
from sklearn.model_selection import train_test_split
from sklearn import svm
import FSVM
import LS_FSVM
from imblearn.over_sampling import SVMSMOTE
import Precision

'''
    Bagging 
    
        n_estimators : integer, the numbre of estimators
        
        estimator : estimator
        
        rate : float, [0,1] , default = 1
            in each estimator, the shape of sub-training_data is rate * trainging_data
            
        methode : optional, 'svm' / 'fsvm' / 'msfsvm'
        
        sampling : optional, 'LowSampling' / 'UpSampling'
                   2 methods to balance dataset, make the numbre of majority class 
                 equale to the minority class

'''
class Bagging(object):
    
    def __init__(self,n_estimators,estimator,rate=1.0, methode='svm',sampling='origine'):
        self.n_estimators = n_estimators
        self.estimator = [estimator for _ in range(self.n_estimators)]
        self.rate = rate
        self.methode = methode
        self.sampling = sampling
    
    def RepetitionRandomSampling(self, x_train, y_train):  # 有放回采样，number为抽样的个数
        x_sample = list()
        y_sample = list()
        for i in range(int(self.rate * len(y_train))):
            index = random.randint(0, len(y_train) - 1)
            x_sample.append(x_train[index])
            y_sample.append(y_train[index])
        return np.array(x_sample),np.array(y_sample)
    
    def MutModel_clf(self, x_train, y_train, x_test):
        y_pred = []

        
        #SVM
        if self.methode == 'svm':
            for estimator in self.estimator:
                x_sample,y_sample = self.RepetitionRandomSampling(x_train, y_train)
                
                if self.sampling=='UpSampling':
                    x_sample, y_sample = SVMSMOTE(random_state=42).fit_sample(x_sample,y_sample)
                elif self.sampling=='LowSampling':
                    train_data = np.concatenate((x_sample,np.array([y_sample]).reshape(len(y_sample),1)),axis=1)
                    data_maj = train_data[np.array(y_sample) == 1]  # 将多数
                    data_min =  train_data[np.array(y_sample) != 1] 
                    index = np.random.randint(len(data_maj), size=len(data_min)) 
                    lower_data_maj = data_maj[list(index)]
                    sample = np.append(lower_data_maj,data_min,axis=0)
                    x_sample = sample[:,:-1]
                    y_sample = sample[:, -1]
                else:
                    x_sample = x_sample
                    y_sample = y_sample
                    
                estimator.fit(x_sample,y_sample)
                y_pred.append(estimator.predict(x_test))
                
                
        #Fuzzy SVM
        if self.methode == 'fsvm':
            for estimator in self.estimator:
                x_sample,y_sample = self.RepetitionRandomSampling(x_train, y_train)
                
                if self.sampling=='UpSampling':
                    x_sample, y_sample = SVMSMOTE(random_state=42).fit_sample(x_sample,y_sample)
                elif self.sampling=='LowSampling':
                    train_data = np.concatenate((x_sample,np.array([y_sample]).reshape(len(y_sample),1)),axis=1)
                    data_maj = train_data[np.array(y_sample) == 1]  # 将多数
                    data_min =  train_data[np.array(y_sample) != 1] 
                    index = np.random.randint(len(data_maj), size=len(data_min)) 
                    lower_data_maj = data_maj[list(index)]
                    sample = np.append(lower_data_maj,data_min,axis=0)
                    x_sample = sample[:,:-1]
                    y_sample = sample[:, -1]
                else:
                    x_sample = x_sample
                    y_sample = y_sample
                estimator._mvalue(x_sample,y_sample)
                estimator.fit(x_sample,y_sample)
                y_pred.append(estimator.predict(x_test))
                
        # LS Fuzzy SVM      
        if self.methode == 'lsfsvm':
            for estimator in self.estimator:
                x_sample,y_sample = self.RepetitionRandomSampling(x_train, y_train)
                
                if self.sampling=='UpSampling':
                    x_sample, y_sample = SVMSMOTE(random_state=42).fit_sample(x_sample,y_sample)
                elif self.sampling=='LowSampling':
                    train_data = np.concatenate((x_sample,np.array([y_sample]).reshape(len(y_sample),1)),axis=1)
                    data_maj = train_data[np.array(y_sample) == 1]  
                    data_min =  train_data[np.array(y_sample) != 1] 
                    index = np.random.randint(len(data_maj), size=len(data_min)) 
                    lower_data_maj = data_maj[list(index)]
                    sample = np.append(lower_data_maj,data_min,axis=0)
                    x_sample = sample[:,:-1]
                    y_sample = sample[:, -1]
                else:
                    x_sample = x_sample
                    y_sample = y_sample

                estimator._mvalue(x_sample,y_sample)
                estimator.fit(x_sample,y_sample)
                y_pred.append(estimator.predict(x_test))
            
            
        result = sum(np.array(y_pred))
        result[result >=1]=1
        result[result <=-1]=-1
            
        return result
    
 
if __name__ == '__main__':
    
    data = DataDeal.get_data()
    Train_data,test = train_test_split(data, test_size=0.2)
        
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
        
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Exp'}
    
    clf = FSVM.FSVM(3, kernel_dict, fuzzyvalue, 3/4)

    bag = Bagging(20, clf, 0.7,'fsvm','UpSampling')
    y_pred= bag.MutModel_clf(x_train,y_train,x_test)
    
    Precision.precision(y_pred,y_test)
