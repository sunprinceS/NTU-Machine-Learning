import numpy as np

FLIPPING_NOISE = 0.2

def read_data(fname):
    arr = np.loadtxt(fname)
    X = arr[:,:-1]
    y = arr[:,-1]
    return X,y

def gen_data(N):
    X = np.random.uniform(-1,1,N)
    y = np.sign(X)
    noise_idx = np.random.choice(N,int(FLIPPING_NOISE*N))

    y[noise_idx] *= -1 #flipping
    return X,y
