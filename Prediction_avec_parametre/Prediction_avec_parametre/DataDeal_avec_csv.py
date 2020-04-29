#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:05:40 2020

@author: nelson
"""
import pandas as pd
import numpy as np

def datadeal(fichier_PCA,fichier_Scaler, X):
    X = np.array(X)
    PCA = pd.read_csv(fichier_PCA)
    scaler_df = pd.read_csv(fichier_Scaler)
    means = PCA.iloc[:,-1]
    means = means.values      # change the type as array
    
    components_T = PCA.iloc[:,:-1]
    components_T = np.array(components_T) # change the type as ndarray
    
    td = X - means   #center columns by subtracting column means
    X_pca = np.dot(td, components_T)  # project data
    
    X_scale = scaler_df.scale_.values * X_pca + scaler_df.min_.values
    
    return X_scale    
    
        
if __name__ == '__main__':
    data = pd.read_csv('Data_numerique.csv')
    df = data.iloc[:,:-1]
    lable = data.iloc[:,-1]
    datadeal('Parameter/PCA.csv','Parameter/Scaler.csv',df)
