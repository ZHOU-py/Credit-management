#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:51:57 2020

@author: nelson
"""

import DataDeal

from sklearn.model_selection import train_test_split
import numpy as np
import LS_FSVM
from sklearn.metrics import roc_auc_score
import Precision


"""
Function LS_FSVM_best

    kernel_dict_type : 'RBF' / 'LINEAR' / 'POLY'
    
    param_grid : dict or list of dictionaries
    
    judgment : 'Acc' / 'AUC',  Default = 'Acc'
            'Acc': judgment methode is the best total accuracy  
            'AUC': judgment methode is the best score of AUC
    
    fuzzyvalue : dict or list of dictionaries, Default = {'type':'Cen','function':'Lin'}
            {'type':'Cen','function':'Lin'} 
            {'type':'Cen','function':'Exp'}
            {'type':'Hyp','function':'Lin'}
            {'type':'Hyp','function':'Exp'}
            
    r_max : float, between 0 and 1 , Default = 1
    r_min : float, between 0 and 1 , Default = 1 
      
      usually for the majority class r = len(y_minority)/len(y_majority) 
          and for the minority class r = 1
    

"""

def LS_FSVM_best(train_data,test,kernel_dict_type, param_grid, judgment='Acc',\
                 fuzzyvalue = {'type':'Cen','function':'Lin'} , r_max=1, r_min=1):   
    
    test_length = test.shape[0]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
            
    score = 0
    best_score = 0
    score_memory = []
    predict_ensemble = []
    for i in range(test_length):
        predict_ensemble.append(0)
    
    if kernel_dict_type == 'RBF':
        for C in param_grid['C']:
            for sigma in param_grid['sigma']:
                kernel_dict = {'type': 'RBF', 'sigma': sigma}
                clf = LS_FSVM.LSFSVM(C, kernel_dict, fuzzyvalue,r_max, r_min)
                clf._mvalue(x_train, y_train)
                clf.fit(x_train, y_train)
                y_predict = clf.predict(x_test)
                if judgment == 'Acc':
                    score = len(y_test[y_predict == y_test]) / test_length
                elif judgment == 'AUC':
                    score = roc_auc_score(y_test, y_predict)
                score_memory.append(score)
                if score > best_score:
                    best_score = score
                    best_parameter = [C, sigma]


    elif kernel_dict_type == 'LINEAR':
        for C in param_grid['C']:
            kernel_dict = {'type': 'LINEAR'}
            clf = LS_FSVM.LSFSVM(C,kernel_dict,fuzzyvalue, r_max, r_min)
            clf._mvalue(x_train, y_train)
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            if judgment == 'Acc':
                score = len(y_test[y_predict == y_test]) / test_length
            elif judgment == 'AUC':
                score = roc_auc_score(y_test, y_predict)
            score_memory.append(score)
            if score > best_score:
                best_score = score
                best_parameter = [C]
        
        
    elif kernel_dict_type == 'POLY':
        for C in param_grid['C']:
            for d in param_grid['d']:
                kernel_dict = {'type': 'POLY','d': d}
                clf = LS_FSVM.LSFSVM(C,kernel_dict, fuzzyvalue, r_max, r_min)
                clf._mvalue(x_train, y_train)
                clf.fit(x_train, y_train)
                y_predict = clf.predict(x_test)
                if judgment == 'Acc':
                    score = len(y_test[y_predict == y_test]) / test_length
                elif judgment == 'AUC':
                    score = roc_auc_score(y_test, y_predict)
                score_memory.append(score)
                if score > best_score:
                    best_score = score
                    best_parameter = [C,d]
                    
    print('kernel_dict:', kernel_dict_type)  
    print('best_parameter',best_parameter)  
    return best_parameter



    
if __name__ == '__main__':
    data = DataDeal.get_data()[:800,:]
    train_data,test = train_test_split(data, test_size=0.2)
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    
    fuzzyvalue = {'type':'Cen','function':'Lin'} 
    param_grid = {'C': np.logspace(0, 1, 50), 'sigma': np.logspace(-2, 0.5, 50)}
    
    C = LS_FSVM_best(train_data,test,'LINEAR',param_grid,'AUC',fuzzyvalue, 3/4, 1)
    kernel_dict = {'type': 'LINEAR'} 
    
    clf = LS_FSVM.LSFSVM(C,kernel_dict,fuzzyvalue,3/4)
    clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    
    Precision.precision(y_predict,y_test)
    
    
    
    
    
    