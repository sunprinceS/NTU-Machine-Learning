import numpy as np

def read_data(fname):
    arr = np.loadtxt(fname)
    X = arr[:,:-1]
    y = arr[:,-1]
    return X,y

def pad_one(X):
    return np.hstack((np.ones((X.shape[0],1)),X))
