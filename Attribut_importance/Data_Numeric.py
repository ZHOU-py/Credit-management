#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:50:00 2020

@author: Zinan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def Data_numerique(X):
    df = []
    for col in X.columns:
        if X[col].dtype == object:
            X_object = X[col]
            encode=OneHotEncoder(sparse = False)
            array_out=encode.fit_transform(np.array(X[col]).reshape(-1,1))
            df_object = pd.DataFrame(array_out, columns=X[col].unique())
            df = pd.concat([pd.DataFrame(df),df_object], axis=1) 
            
        else :
           # print(col)
            X_object = X[col]
            df = pd.concat([pd.DataFrame(df),pd.DataFrame(X_object)], axis=1)
    
#    print(len(df.columns))
    return df           
    
    

if __name__ == '__main__':
    data = pd.read_csv('german_credit.csv')
#    print(data.describe())
    X = data.drop(['default'],axis = 1)
    df = Data_numerique(X)
    df = pd.concat([df,pd.DataFrame(data['default'])],axis=1)
#    print(df)