import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
'''
def get_data():
    df = pd.read_csv('german.csv',header=None, quoting=3)

    data = np.array(df)

    #data[:,:-1] = preprocessing.scale(data[:,:-1])
    #print(data.shape)
    data[:, -1][data[:, -1] == 2] = -1


    return data

'''
"""
def get_data():
    df = pd.read_csv('australian.csv',header=None)

    data = np.array(df)
    data[:, :-1] = preprocessing.scale(data[:, :-1])
    data[:,-1][data[:,-1]==0] = -1

    return data
"""

def get_data():
    df = pd.read_csv('german_numerical.csv',header=None, quoting=3)
    df = df.drop([0],axis=0)
    df = df.astype('int64')
    data = np.array(df)

    #data[:,:-1] = preprocessing.scale(data[:,:-1])
    #print(data.shape)
    data[:, -1][data[:, -1] == 2] = -1
    X = pd.DataFrame(data[:, :-1])
    pca = PCA(n_components = 30 ) # réserver 3 compinents
    X = np.array(X)
    pca.fit(X)
    # Le pourcentage de variation de toutes les fonctionnalités est de 
    # [9.99971458e-01 1.64276254e-05 1.09629804e-05] 
    # Signifie que presque toutes les informations sont conservées
    
    #print(pca.explained_variance_ratio_)
    
    # les variances de ces 3 components sont
    # [7.96790062e+06 1.30897422e+02 8.73544315e+01]
    
    #print(pca.explained_variance_) 
    
    X_pca = pca.transform(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_pca = min_max_scaler.fit_transform(X_pca)
    data = np.append(X_pca,data[:, -1:],axis=1)
    
    return data

#data = get_data()
#print(len(data[data[:,-1] == -1]))
#print(get_data())