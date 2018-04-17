import numpy as np

FLIPPING_NOISE = 0.1

def pad_one(X):
    return np.hstack((np.ones((X.shape[0],1)),X))

def read_data(fname):
    arr = np.loadtxt(fname)
    X = arr[:,:-1]
    y = arr[:,-1]
    return X,y

def feat_transform(X):
    """
    General Quadratic Feature Transform
    """
    N = X.shape[0]
    x1x2 = np.apply_along_axis(lambda x: x[1]*x[2],1,X)
    return np.hstack((X,x1x2.reshape(N,1),(X[:,1]**2).reshape(N,1),(X[:,2]**2).reshape(N,1)))

def gen_data(N):
    X = np.random.uniform(-1,1,(N,2))
    y = np.apply_along_axis(lambda x: np.sign(x[0]**2 + x[1]**2 - 0.6),1,X)

    noise_idx = np.random.choice(N,int(FLIPPING_NOISE*N))
    y[noise_idx] *= -1 #flipping

    return X,y
