#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:31:17 2020

@author: nelson
"""
import numpy as np
import pandas as pd
import Data_Numeric
import DataDeal
from sklearn.model_selection import train_test_split
import Precision


def predict(fichier,fichier_Xtrain, Xtest):
    
    df = pd.read_csv(fichier)
    Xtrain = pd.read_csv(fichier_Xtrain)
    Xtrain = np.array(Xtrain)
    A = np.multiply(df.alpha, df.Y)
    
    if df.Kernel[0]=='RBF':
        X2_train = np.sum(np.multiply(Xtrain, Xtrain), 1)
        X2_test = np.sum(np.multiply(Xtest, Xtest), 1)
        tmp = np.matrix(X2_train) + np.matrix(X2_test).T
        if tmp.shape[0] != X2_test.shape[0]:
            tmp = tmp.T
        K0 = tmp - 2 * np.dot(Xtest, Xtrain.T)   
        testMat = np.array(np.power(np.exp(-1.0 / (2* df['K.sigma'][0] ** 2)), K0))
        y_predict = df.b[0] + np.dot(testMat, A)
    elif df.Kernal[0]=='POLY':
        testMat = np.power((np.dot(Xtest, Xtrain.T) + 1), df['K.d'][0])
        y_predict = df.b[0] + np.dot(testMat, A)
    elif df.Kernel[0]=='LINEAR':
        testMat = np.dot(Xtest, Xtrain.T)
        y_predict = df.b[0] + np.dot(testMat , A)
        
    y_pred = np.sign(y_predict)
    
    return y_pred


def predict_prob(fichier,fichier_Xtrain, Xtest):
    df = pd.read_csv(fichier)
    Xtrain = pd.read_csv(fichier_Xtrain)
    Xtrain = np.array(Xtrain)
    A = np.multiply(df.alpha, df.Y)
    
    if df.Kernel[0]=='RBF':
        X2_train = np.sum(np.multiply(Xtrain, Xtrain), 1)
        X2_test = np.sum(np.multiply(Xtest, Xtest), 1)
        tmp = np.matrix(X2_train) + np.matrix(X2_test).T
        if tmp.shape[0] != X2_test.shape[0]:
            tmp = tmp.T
        K0 = tmp - 2 * np.dot(Xtest, Xtrain.T)   
        testMat = np.array(np.power(np.exp(-1.0 / (2* df['K.sigma'][0] ** 2)), K0))
        y_predict = df.b[0] + np.dot(testMat, A)
    elif df.Kernal[0]=='POLY':
        testMat = np.power((np.dot(Xtest, Xtrain.T) + 1), df['K.d'][0])
        y_predict = df.b[0] + np.dot(testMat, A)
    elif df.Kernel[0]=='LINEAR':
        testMat = np.dot(Xtest, Xtrain.T)
        y_predict = df.b[0] + np.dot(testMat , A)

    y_prob = 1/(1+np.exp( df.A[0]*y_predict+df.B[0]))
    
    for i in range(len(y_prob)):
        y_prob[i] = round(y_prob[i],3)
        
    return y_prob


# Test Code 
if __name__ == '__main__':
    
    data = pd.read_csv('german_credit.csv')
    #    print(data.describe())
    X = data.drop(['default'],axis = 1)
    lable = data['default']
    df = Data_Numeric.Data_numerique(X)
    data = DataDeal.get_data(df,lable)
    #data = pd.read_csv('german_numerical.csv')
    #data = DataDeal.get_data(data.iloc[:,:-1],data.iloc[:,-1])
    
    x = data[:,:-1]
    y = data[:,-1]
        
    Train_data,test = train_test_split(data, test_size=0.2,random_state = 42)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]

    y_pred = predict('Parameter/LSFSVM_Cen_Lin_RBF_Origine.csv', 'Parameter/X_train.csv',x_test)
    y_prob = predict_prob('Parameter/LSFSVM_Cen_Lin_RBF_Origine.csv', 'Parameter/X_train.csv',x_test)
#    y_pred = predict('Parameter/FSVM_Cen_Lin_RBF_Origine.csv', 'Parameter/X_train.csv',x_test)
#    y_prob = predict_prob('Parameter/FSVM_Cen_Lin_RBF_Origine.csv', 'Parameter/X_train.csv',x_test)
    
 #   print(y_pred)
 #   print(y_prob)
    
    Precision.precision(y_pred,y_test)