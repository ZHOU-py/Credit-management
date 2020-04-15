import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def get_data(df, lable):
    
    df = df.astype('int64')
    data = np.array(df)
    lable = lable.values
    #data[:,:-1] = preprocessing.scale(data[:,:-1])
    #print(data.shape)
    lable[lable == 0] = -1
    X = pd.DataFrame(data[:, :-1])
    pca = PCA(n_components = 30) # réserver 3 compinents

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
    data = np.append(X_pca,lable[:,None],axis=1)
    
    return data

if __name__ == '__main__':
    data = pd.read_csv('german_numerical.csv')
    df = data.drop(['class'],axis = 1)
    lable = data['class']
    data = get_data(data,lable)
    Train_data,test = train_test_split(data, test_size=0.2)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]



