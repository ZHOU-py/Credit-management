#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:34:34 2020

@author: nelson
"""
from random import randrange
import DataDeal
from sklearn import svm
from sklearn.model_selection import train_test_split
import Precision
import numpy as np
import LS_FSVM
import FSVM
from imblearn.over_sampling import SVMSMOTE
import pandas as pd
import csv

def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def LS_FSVM_bagging(data):   
    train_data,test = train_test_split(data, test_size=0.2)
    test_length = test.shape[0]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    y_train = train_data[:,-1]

    predict_ensemble_ls = []
    predict_ensemble_f = []
    
    for i in range(test_length):
        predict_ensemble_ls.append(0)
        
    for i in range(test_length):
        predict_ensemble_f.append(0)
    
    
    for i in range(7):
        sample = np.array(subsample(dataset=data[:-test_length, :], ratio=0.3))
    
        train_data = sample
    
        train_data = pd.DataFrame(train_data)
        data_maj = train_data[train_data.iloc[:,-1] == 1]  # 将多数
        data_min = train_data[train_data.iloc[:,-1] != 1] 
        index = np.random.randint(len(data_maj), size=len(data_min)) 
        lower_data_maj = data_maj.iloc[list(index)]
        train_data = np.asarray(pd.concat([lower_data_maj, data_min]))
        y_train = train_data[:, -1]
        
        score = 0
        best_score = 0
        score_memory = []
        predict_ensemble = []
        for i in range(test_length):
            predict_ensemble.append(0)
            
        for C in np.logspace(0, 3, 50):
            for gamma in np.logspace(0, 2, 25):
                kernel_dict = {'type': 'RBF', 'gamma': gamma}
                (alpha, b, K) = LS_FSVM._LSSVMtrain(train_data, kernel_dict, C)
                y_predict = LS_FSVM._LSSVMpredict(x_test, K, alpha, b, y_train)
                score = len(y_test[y_predict == y_test]) / test_length
                score_memory.append(score)
                if score > best_score:
                    best_score = score
                    best_parameter = [C, gamma]
                    
        gamma = best_parameter[1]
        kernel_dict = {'type': 'RBF', 'gamma': gamma}
        (alpha, b, K) = LS_FSVM._LSSVMtrain(train_data, kernel_dict, best_parameter[0])
        y_predict = LS_FSVM._LSSVMpredict(x_test, K, alpha, b, y_train)
        predict_ensemble_ls = predict_ensemble_ls + y_predict
        print(predict_ensemble_ls)
        
        b_ls = b
#        alpha_ls = alpha.copy()
        #print('LS-FSVM b:', b_ls)
        #print('LS-FSVM alpha:',alpha_ls)    
        bad_ls, good_ls,type1_ls,type2_ls,score_ls,auc_ls = Precision.precision(y_predict,y_test)
        
        scale = np.sqrt(1/gamma)
        kernel_dict = {'type': 'RBF', 'gamma': scale}
        (alpha, b, K,X,Y) = FSVM._FSVMtrain(train_data, kernel_dict, best_parameter[0])
        Y_predict = FSVM._FSVMpredict(x_test, kernel_dict, alpha, b, X, Y)
        predict_ensemble_f = predict_ensemble_f + Y_predict
        print(predict_ensemble_f)
        
        b_f = b
#        alpha_f = alpha.copy()
        #print('FSVM b:',b_f)
        #print('FSVM alpha:', alpha_f) 
        bad_f, good_f, type1_f,type2_f,score_f,auc_f = Precision.precision(Y_predict,y_test)
      
        

    predict_ensemble_ls[predict_ensemble_ls >= 1] = 1
    predict_ensemble_ls[predict_ensemble_ls <= -1] = -1
    
    predict_ensemble_f[predict_ensemble_f >= 1] = 1
    predict_ensemble_f[predict_ensemble_f <= -1] = -1
    
    
    bad_ls, good_ls,type1_ls,type2_ls,score_ls,auc_ls = Precision.precision(predict_ensemble_ls,y_test)
    bad_f, good_f, type1_f,type2_f,score_f,auc_f = Precision.precision(predict_ensemble_f,y_test)

    LSFSVM = {'bad_ls':[bad_ls],'good_ls':[good_ls],\
              'type1_ls':[type1_ls], 'type2_ls': [type2_ls],\
              'total accuracy_ls': [score_ls],'AUC_ls': [auc_ls],'b_ls':[b_ls]}
    
    
    FSVM1 = {'bad_f':[bad_f],'good_ls':[good_f],\
              'type1_f':[type1_f], 'type2_f': [type2_f],\
              'total accuracy_f': [score_f],'AUC_ls': [auc_f],'b_f':[b_f]}
    
#    pd.DataFrame(data=LSFSVM).to_csv("Precision/LSFsvm_bagging_lowSampling.csv",index=0)
#    pd.DataFrame(data=FSVM1).to_csv("Precision/Fsvm_bagging_lowSampling.csv",index=0)

   
    path  = "Precision/LSFsvm_bagging_lowSampling.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [bad_ls, good_ls, type1_ls,type2_ls,score_ls,auc_ls,b_ls]
        csv_write.writerow(data_row)

    path  = "Precision/Fsvm_bagging_lowSampling.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [bad_f, good_f, type1_f,type2_f,score_f,auc_f,b_f]
        csv_write.writerow(data_row)

        
if __name__ == '__main__':
    data = DataDeal.get_data() 
    for i in range(3):
        print(i)
        LS_FSVM_bagging(data)        
        
        
        