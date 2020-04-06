#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:53:31 2020

@author: nelson
"""

import DataDeal
from random import randrange
from sklearn.model_selection import train_test_split
import numpy as np
import FSVM
import GridSearch_parametre
import Precision
from imblearn.over_sampling import SVMSMOTE

'''
    FSVM Bagging
    
        data : ndarry, will be seperated into training part and test part
        
        databalance: optional, 'LowSampling' / 'UpSampling'
                   2 methods to balance dataset, make the numbre of majority class 
                 equale to the minority class
    
        kernel_dict_type : optional, 'RBF' / 'LINEAR' / 'POLY'
                'RBF' : chose the best paramatre C and sigma by gridesearch
                'LINEAR' : chose the best paramatre C by gridesearch
                'POLY' : chose the best paramatre C and d by gridesearch
                
        param_grid : the grids spanned by each dictionary in the list are explored. 
         
        judgment : optional, 'Acc' / 'AUC'
                The criteria for selecting the best parameters
                
        fuzzyvalue:
              membershape value based on the class of center
              'type': 'Cen' 
              'function' : 'Lin' / 'Exp'
              
              membershape value based on the actuale hyper-plane
              'type': 'Hyp' 
              'function' : 'Lin' / 'Exp'
              
              r_max : radio between 0 and 1
              r_min : radio between 0 and 1    for balancing data
              
              usually for the majority class r = len(y_minority)/len(y_majority) 
                  and for the minority class r = 1

'''

def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample
        
        
def FSVM_bagging(data, databalance, kernel_dict_type,  param_grid, judgment='Acc',\
                 fuzzyvalue = {'type':'Cen','function':'Lin'} , r_max=1, r_min=1):
    
    train_data,test = train_test_split(data, test_size=0.2)
    test_length = test.shape[0]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    y_train = train_data[:,-1]

    predict_ensemble = []
    
    for i in range(test_length):
        predict_ensemble.append(0)

    for i in range(7):
        sample = np.array(subsample(dataset=train_data, ratio=0.8))
        train_data = sample
        x_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        
        if databalance =='LowSampling':
            data_maj = train_data[y_train == 1]  # 将多数
            data_min =  train_data[y_train != 1] 
            index = np.random.randint(len(data_maj), size=len(data_min)) 
            lower_data_maj = data_maj[list(index)]
            train_data = np.append(lower_data_maj,data_min,axis=0)
            x_train = train_data[:,:-1]
            y_train = train_data[:, -1]
    
        elif databalance =='UpSampling':
            x_train, y_train = SVMSMOTE(random_state=42).fit_sample(train_data[:, :-1],\
                                       np.asarray(train_data[:, -1]))
            train_data = np.append(x_train,y_train.reshape(len(y_train),1),axis=1)
            
        else:
            x_train = train_data[:,:-1]
            y_train = train_data[:,-1]
        
        if kernel_dict_type=='LINEAR':
            C = GridSearch_parametre.LS_FSVM_best(train_data,test,kernel_dict_type,\
                                                  param_grid,judgment,fuzzyvalue, r_max, r_min)
            kernel_dict={'type': 'LINEAR'}
            
        elif kernel_dict_type=='RBF':
            C,sigma = GridSearch_parametre.LS_FSVM_best(train_data,test,kernel_dict_type,\
                                                  param_grid,judgment,fuzzyvalue, r_max, r_min)
            kernel_dict={'type': 'RBF','sigma':sigma}
            
        elif kernel_dict_type=='POLY':
            C,d = GridSearch_parametre.LS_FSVM_best(train_data,test,kernel_dict_type,\
                                                  param_grid,judgment,fuzzyvalue, r_max, r_min)
            kernel_dict={'type': 'POLY','d': d}
            
        clf = FSVM.FSVM(C,kernel_dict,fuzzyvalue,r_max, r_min)
        clf._mvalue(x_train, y_train)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        
        predict_ensemble = predict_ensemble + y_predict
        Precision.precision(y_predict,y_test)
        print(predict_ensemble)
       
    predict_ensemble[predict_ensemble >= 1] = 1
    predict_ensemble[predict_ensemble <= -1] = -1
    
    Precision.precision(predict_ensemble,y_test)
        
if __name__ == '__main__':
    data = DataDeal.get_data() 
    fuzzyvalue = {'type':'Cen','function':'Lin'} 
    param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1, 0, 30)}
#    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
    databalance = 'UpSampling'
    FSVM_bagging(data,databalance,'RBF',param_grid,'Acc',fuzzyvalue, 3/4)  
