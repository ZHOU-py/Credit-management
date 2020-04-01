#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:59:22 2020

@author: nelson
"""
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


#SVM
def SVM_best(data):
    train_data,test = train_test_split(data, test_size=0.2)
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
        
    for C in [0.001, 0.01, 0.1, 1, 10]:
        for gamma in [0.001, 0.01, 0.1, 1]:
            clf = svm.SVC(kernel='rbf')
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            
            score = len(y_test[y_predict == y_test]) / test_length
            score_memory.append(score)
            if score > best_score:
                best_score = score
                best_parameter = [C, gamma]
                
    clf = svm.SVC(kernel='rbf',C=best_parameter[0],gamma= best_parameter[1])
    clf.fit(x_train, y_train)
    Y_predict = clf.predict(x_test)
    
    Precision.precision(Y_predict,y_test)
   

"""  
#SVM
clf = svm.SVC(kernel='rbf',C=3, gamma='scale')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
Precision.precision(Y_predict,y_test)


scale = 1 / (x_train.shape[0] * x_train.var())

#LS-FSVM
gamma = np.sqrt(1/scale)
kernel_dict = {'type': 'RBF', 'gamma': gamma}

(alpha, b, K) = LS_FSVM_origine._LSSVMtrain(train_data, kernel_dict, 3)
Y_predict = LS_FSVM_origine._LSSVMpredict(x_test, K, alpha, b, y_train)
#print(y_test)
Precision.precision(Y_predict,y_test)


#FSVM 
scale = 1 / (x_train.shape[0] * x_train.var())
kernel_dict = {'type': 'RBF', 'gamma': scale}

(alpha, b, K,X,Y) = FSVM._FSVMtrain(train_data, kernel_dict, 3)
Y_predict = FSVM._FSVMpredict(x_test, kernel_dict, alpha, b, X, Y)
#print(y_test)
Precision.precision(Y_predict,y_test)
"""
#LS_FSVM

def LS_FSVM_best(data):   
    train_data,test = train_test_split(data, test_size=0.2)
    test_length = test.shape[0]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    y_train = train_data[:,-1]
        
    
    score = 0
    best_score = 0
    score_memory = []
    predict_ensemble = []
    for i in range(test_length):
        predict_ensemble.append(0)
        
    for C in np.logspace(0, 3, 50):
#        for gamma in np.logspace(0, 2, 25):
        for d in range(10):
            #kernel_dict = {'type': 'RBF', 'gamma': gamma}
            #kernel_dict = {'type': 'LINEAR'}
            kernel_dict = {'type': 'POLY','d': d}
            (alpha, b, K) = LS_FSVM._LSSVMtrain(train_data, kernel_dict, C)
            y_predict = LS_FSVM._LSSVMpredict(x_test, K, alpha, b, y_train)
            score = len(y_test[y_predict == y_test]) / test_length
            score_memory.append(score)
            if score > best_score:
                best_score = score
#                best_parameter = [C, gamma]
#                best_parameter = [C]
                best_parameter = [C,d]
                
#    gamma = best_parameter[1]
    d = best_parameter[1]
#    kernel_dict = {'type': 'RBF', 'gamma': gamma}
#    kernel_dict = {'type': 'LINEAR'}
    kernel_dict = {'type': 'POLY','d': d}
    (alpha, b, K) = LS_FSVM._LSSVMtrain(train_data, kernel_dict, best_parameter[0])
    y_predict = LS_FSVM._LSSVMpredict(x_test, K, alpha, b, y_train)
    b_ls = b
#    alpha_ls = alpha.copy()
    #print('LS-FSVM b:', b_ls)
    #print('LS-FSVM alpha:',alpha_ls)    
    bad_ls, good_ls,type1_ls,type2_ls,score_ls,auc_ls = Precision.precision(y_predict,y_test)
    
#    scale = np.sqrt(1/gamma)
#    kernel_dict = {'type': 'RBF', 'gamma': scale}
#    kernel_dict = {'type': 'LINEAR'}
    kernel_dict = {'type': 'POLY','d': d}
    membership = FSVM.memership_value(train_data)
    (alpha, b, K,X,Y) = FSVM._FSVMtrain(train_data, kernel_dict, best_parameter[0],membership)
    Y_predict = FSVM._FSVMpredict(x_test, kernel_dict, alpha, b, X, Y)
    b_f = b
#    alpha_f = alpha.copy()
    #print('FSVM b:',b_f)
    #print('FSVM alpha:', alpha_f) 
    bad_f, good_f, type1_f,type2_f,score_f,auc_f = Precision.precision(Y_predict,y_test)
    
    LSFSVM = {'bad_ls':[bad_ls],'good_ls':[good_ls],\
              'type1_ls':[type1_ls], 'type2_ls': [type2_ls],\
              'total accuracy_ls': [score_ls],'AUC_ls': [auc_ls],'b_ls':[b_ls]}
    
    
    FSVM1 = {'bad_f':[bad_f],'good_ls':[good_f],\
              'type1_f':[type1_f], 'type2_f': [type2_f],\
              'total accuracy_f': [score_f],'AUC_ls': [auc_f],'b_f':[b_f]}
    
#    pd.DataFrame(data=LSFSVM).to_csv("Distance/LSFsvm_cen_lin.csv",index=0)

#    pd.DataFrame(data=FSVM1).to_csv("Distance/Fsvm_cen_lin.csv",index=0)

    path  = "Kernel/LSFsvm_poly.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [bad_ls, good_ls, type1_ls,type2_ls,score_ls,auc_ls,b_ls]
        csv_write.writerow(data_row)

    path  = "Kernel/Fsvm_poly.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [bad_f, good_f, type1_f,type2_f,score_f,auc_f,b_f]
        csv_write.writerow(data_row)

'''
    path  = "Precision/LSFsvm_alpha.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = alpha_ls
        csv_write.writerow(data_row)

    path  = "Precision/Fsvm_alpha.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = alpha_f
        csv_write.writerow(data_row)
'''
if __name__ == '__main__':
    data = DataDeal.get_data() 
    for i in range(11):
        print(i)
        LS_FSVM_best(data)
    
    
    
    
    
    