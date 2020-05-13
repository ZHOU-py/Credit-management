import pandas as pd
import numpy as np
import random
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import Precision


def get_data(df, lable,processing='normaliser',scaler='False'):
    
    df = df.astype('int64')
    X = np.array(df)
    lable = lable.values

    if processing == 'scaler':            
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
    
    elif processing == 'normaliser':
        X = (X - X.mean())/X.std()
    
    pca = PCA(n_components = 36) # réserver 3 compinents    
    pca.fit(X)    
    X_pca = pca.transform(X)
    
    if scaler=='True':        
        min_max_scaler = preprocessing.MinMaxScaler()
        X_pca = min_max_scaler.fit_transform(X_pca)        
        data = np.append(X_pca,lable[:,None],axis=1)
    else:
        data = np.append(X_pca,lable[:,None],axis=1)
    
    return data

if __name__ == '__main__':
    
    data = pd.read_csv("data/Database_Encodage.csv")
#    data = pd.read_csv("data/Database_label.csv")
#    data = pd.read_csv("data/Database_onehotencoder.csv")
    X = data.drop(['Loan classification'],axis = 1)
    label = data['Loan classification']
    data = get_data(X,label,'scaler','True')
#    data = get_data(X,label,'scaler','False')
    Train_data,test = train_test_split(data, test_size=0.2,random_state = 42)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
    
    clf = svm.SVC(C = 3,class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)
    
    clf = svm.LinearSVC(C=3,class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)








