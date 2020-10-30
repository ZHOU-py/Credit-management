#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:07:25 2020

@author: nelson
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import Precision
import Improved_RBF_model

    
if __name__ == '__main__':
    data = pd.read_csv("new_data/Data.csv")
    
    X = data.drop(['Loan classification'],axis = 1)
    x = data.drop(['Loan classification'],axis = 1)
    label = data['Loan classification']
    
    X1 = (X - X.mean())/X.std()
    dataframe = pd.DataFrame({'Feature':X.columns,'Mean':X.mean(),'Std':X.std()})
    
    
    X1 = X1.astype('float64')
    X1 = np.array(X1)
    data['Loan classification'][data['Loan classification']==1] = -1
    data['Loan classification'][data['Loan classification']==0] = 1
    
    lable = data['Loan classification'].copy().values
    data = np.append(X1,data['Loan classification'][:,None],axis=1)
    
    print('Clients %s'%Counter(lable))
        
    PrecisionArray,PrecisionArray1 = [],[]
    
    for i in range(1):
    
        Train_data,test = train_test_split(data, test_size=0.01,shuffle = True)
        #Train_data,test = train_test_split(data, test_size=0.3,shuffle = True,random_state = 10)
        x_test = test[:,:-1]
        x_test = x_test.astype('float')
        y_test = test[:,-1]
        y_test=y_test.astype('int')
        x_train = Train_data[:,:-1]
        x_train = x_train.astype('float')
        y_train = Train_data[:,-1]
        y_train=y_train.astype('int')
                
        threshold1_low = 0.3
        threshold2_low = 0.35
        threshold3_low = 0.5
        
#        for threshold1_up,threshold2_up in zip([0.8,0.75,0.7],[0.75,0.7,0.65]):
#            print('threshold1_up,threshold2_up',threshold1_up,threshold2_up)
        threshold1_up = 0.7
        threshold2_up = 0.65
        threshold3_up = 0.5
        
                                 
#        y_pred2, y_Pred = Improved_RBF_model.Improved_BRF_low(x_train,y_train,x_test,y_test,\
#                                                                threshold1_low,threshold2_low,threshold3_low)
#        y_pred2, y_Pred = Improved_RBF_model.Improved_BRF_up(x_train,y_train,x_test,y_test,\
#                                                                threshold1_up,threshold2_up,threshold3_up)
        y_pred2, y_Pred = Improved_RBF_model.Improved_BRF(x_train,y_train,x_test,y_test,\
                                                             threshold1_up,threshold2_up,threshold3_up,\
                                                             threshold1_low,threshold2_low,threshold3_low,)
#            Precision.precision(y_pred2,y_test)
#            Precision.precision(y_Pred,y_test)
        PrecisionArray.append(Precision.precision(y_pred2,y_test))
        PrecisionArray1.append(Precision.precision(y_Pred,y_test))
    
    
    print('********* Banch Mark *********')
    banchmark = np.round(np.mean(np.array(PrecisionArray),axis=0),3)
    print(banchmark)
    
    print('********* Model ********')
    model = np.round(np.mean(np.array(PrecisionArray1),axis=0),3)
    print(model)
    
    print(pd.DataFrame(data = PrecisionArray).describe())
    print(pd.DataFrame(data = PrecisionArray1).describe())

