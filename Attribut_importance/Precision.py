#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:16:13 2020

@author: nelson
"""
from sklearn.metrics import roc_auc_score

def precision(predict_ensemble,y_test):
    predict_ensemble[predict_ensemble >= 1] = 1
    predict_ensemble[predict_ensemble <= -1] = -1
#    print(predict_ensemble)
    
    test_length = len(y_test)
    score = len(y_test[predict_ensemble == y_test]) / test_length
    ob1 = len(predict_ensemble[predict_ensemble == -1])
    ob1_clf1 = 0
    for i in range(len(predict_ensemble)):
        if predict_ensemble[i] == -1 and y_test[i] == -1:
            ob1_clf1 += 1
    type1 = ob1_clf1 / ob1
    
    ob2 = len(predict_ensemble[predict_ensemble == 1])
    ob2_clf2 = 0
    for i in range(len(predict_ensemble)):
        if predict_ensemble[i] == 1 and y_test[i] == 1:
            ob2_clf2 += 1
    type2 = ob2_clf2 / ob2
    
    bad_preci = len(y_test[y_test == -1])
    bad_preci_clf1 = 0
    for i in range(len(y_test)):
        if predict_ensemble[i] == -1 and y_test[i] == -1:
            bad_preci_clf1 += 1
    bad = bad_preci_clf1 / bad_preci
    
    good_preci = len(y_test[y_test == 1])
    good_preci_clf2 = 0
    for i in range(len(y_test)):
        if predict_ensemble[i] == 1 and y_test[i] == 1:
            good_preci_clf2 += 1
    good = good_preci_clf2 / good_preci
    
    auc = roc_auc_score(y_test, predict_ensemble)
    print('bad precision',round(bad,3),'good precision',round(good,3),\
          'type1', round(type1,3), 'type2', round(type2,3), 'total accuracy', round(score,3))
    print('AUC:', auc)
    return (round(bad,3),round(good,3),round(type1,3),round(type2,3),round(score,3),auc)

if __name__ == '__main__':
    precision(predict_ensemble_svm,y_test)