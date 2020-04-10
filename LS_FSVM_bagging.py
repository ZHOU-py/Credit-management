#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:52:40 2020

@author: Zinan
"""

import DataDeal
from random import randrange
import numpy as np
import LS_FSVM
import GridSearch_parametre
import Precision
from imblearn.over_sampling import SVMSMOTE
import pickle
from sklearn.model_selection import train_test_split

'''
    LS-FSVM Bagging
    
    Parameter
    
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

    Methods
         
        fit(self, X, Y)
            Fit the model according to the given training data.
        
        predict(self, X)
            Predict class labels for samples in X.
            
        
        predict_prob(self,X)
            Posterior class probability Pr(y = 1|x)
 
'''


class LS_FSVM_bagging(object):
    
    def __init__(self,n_estimator=3, databalance='originr', kernel_dict_type='RBF',\
                 param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,10,25)},\
                 judgment='Acc',fuzzyvalue = {'type':'Cen','function':'Lin'} , r_max=1, r_min=1):
        
        self.n_estimator = n_estimator
        self.databalance = databalance
        self.kernel_dict_type = kernel_dict_type
        self.param_grid = param_grid
        self.judgment = judgment
        self.fuzzyvalue = fuzzyvalue
        self.r_max = r_max
        self.r_min = r_min

    def subsample(self, dataset, ratio=1.0):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample
        
        
    def fit(self,X,y):
        
        train_data = np.append(X,y.reshape(len(y),1),axis=1)   
        
        clf = [[]]*self.n_estimator
        
        if databalance =='LowSampling':
            data_maj = train_data[y == 1]  # 将多数
            data_min =  train_data[y != 1] 
            index = np.random.randint(len(data_maj), size=len(data_min)) 
            lower_data_maj = data_maj[list(index)]
            train_data = np.append(lower_data_maj,data_min,axis=0)
        
        elif databalance =='UpSampling':
            x_train, y_train = SVMSMOTE(random_state=42).fit_sample(train_data[:, :-1],\
                                       np.asarray(train_data[:, -1]))
            train_data = np.append(x_train,y_train.reshape(len(y_train),1),axis=1)
            
        else:
            train_data = train_data
    
        for i in range(self.n_estimator):
            #sample = np.array(subsample(dataset=data[:-test_length, :], ratio=0.7))
            sample = np.array(self.subsample(dataset=train_data, ratio=0.8))
            train_data = sample        
            x_train = train_data[:,:-1]
            y_train = train_data[:,-1]
            
            if self.kernel_dict_type=='LINEAR':
                C = GridSearch_parametre.LS_FSVM_best(x_train,y_train,self.kernel_dict_type,\
                                                      self.param_grid,self.judgment,self.fuzzyvalue, self.r_max, self.r_min)
                kernel_dict={'type': 'LINEAR'}
                
            elif self.kernel_dict_type=='RBF':
                C,sigma = GridSearch_parametre.LS_FSVM_best(x_train,y_train,self.kernel_dict_type,\
                                                      self.param_grid,self.judgment,self.fuzzyvalue, self.r_max, self.r_min)
                kernel_dict={'type': 'RBF','sigma':sigma}
                
            elif self.kernel_dict_type=='POLY':
                C,d = GridSearch_parametre.LS_FSVM_best(x_train,y_train,self.kernel_dict_type,\
                                                      self.param_grid,self.judgment,self.fuzzyvalue, self.r_max, self.r_min)
                kernel_dict={'type': 'POLY','d': d}
            
            
            clf[i] = LS_FSVM.LSFSVM(C,kernel_dict,self.fuzzyvalue,'origine', self.r_max, self.r_min)
            clf[i]._mvalue(x_train, y_train)
            clf[i].fit(x_train, y_train)
            
            
        with open('LSFsvm_bagging.pkl', 'wb') as f:
            for i in range(self.n_estimator):
                pickle.dump(clf[i], f, pickle.HIGHEST_PROTOCOL) 
                
        
    def predict(self, X):
        test_length = len(X)
        predict_ensemble = []
        for i in range(test_length):
            predict_ensemble.append(0) 
        
        clf = [[]]*self.n_estimator
    

        with open('LSFsvm_bagging.pkl', 'rb') as f:
            for i in range(self.n_estimator):
                clf[i] = pickle.load(f)

        for i in range(self.n_estimator):       
            y_predict = clf[i].predict(X)
            predict_ensemble = predict_ensemble + y_predict           
       
        predict_ensemble[predict_ensemble >= 1] = 1
        predict_ensemble[predict_ensemble <= -1] = -1
        
        return predict_ensemble
    
    def predict_prob(self, X):
        test_length = len(X)
        proba = []
        for i in range(test_length):
            proba.append(0) 
        
        clf = [[]]*self.n_estimator
    

        with open('LSFsvm_bagging.pkl', 'rb') as f:
            for i in range(self.n_estimator):
                clf[i] = pickle.load(f)

        for i in range(self.n_estimator):
            clf[i].predict(X)
            y_prob = clf[i].predict_prob(X)
            proba = proba + y_prob
            
        proba = proba /self.n_estimator
       
        return  proba
        
        
if __name__ == '__main__':
    data = DataDeal.get_data('german_numerical.csv')
    Train_data,test = train_test_split(data, test_size=0.2)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
    
    fuzzyvalue = {'type':'Hyp','function':'Exp'} 
#    param_grid = {'C': np.logspace(0, 2, 40), 'sigma': np.logspace(-1,10,25)}
    param_grid = {'C': np.logspace(0, 1, 50), 'd': range(10)}
    databalance = 'UpSampling'
    lsb = LS_FSVM_bagging(5,databalance,'LINEAR',param_grid,'Acc', fuzzyvalue, 3/4,1) 
    lsb.fit(x_train,y_train)
    predict_ensemble = lsb.predict(x_test)
    prob = lsb.predict_prob(x_test)
    
    print(prob)
    print(predict_ensemble[prob<0.5])
    i = predict_ensemble[prob>0.5]
    y_p = prob[prob>0.5]
    print(y_p[i==-1])
    Precision.precision(predict_ensemble,y_test)



      