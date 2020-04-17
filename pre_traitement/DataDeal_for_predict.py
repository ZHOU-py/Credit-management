#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:02:45 2020

@author: nelson
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def get_data(fichier_name):
    df = pd.read_csv('german_pre.csv',header=None, quoting=3)
    df = df.drop([0],axis=0)
    df = df.astype('int64')
    data = np.array(df)

    #data[:,:-1] = preprocessing.scale(data[:,:-1])
    #print(data.shape)
    data[:, -1][data[:, -1] == 2] = -1
    X = pd.DataFrame(data[:, :-1])
    X = np.array(X)
    with open('PCA.pickle', 'rb') as f:
        pca = pickle.load(f)
        #测试读取后的Model
    X_pca = pca.transform(X)

    with open('MinMaxScaler.pickle', 'rb') as f:
        MinMaxScaler = pickle.load(f)
        
    X_pca = MinMaxScaler.fit_transform(X_pca)

    data = np.append(X_pca,data[:, -1:],axis=1)
    
    return data

if __name__ == '__main__':
    
    data = get_data('german_pre.csv')
    Train_data,test = train_test_split(data, test_size=0.2)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]