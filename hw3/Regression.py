import numpy as np

INF = int(1e9)
EPS = 1e-6

def sigmoid(s):
    return 1/(1+np.exp(-1*s))

class LinearReg(object):

    def __init__(self,dim):
        self.w = np.zeros(dim)

    def fit(self,X,y):
        """
            X: np.array with size N * dim, N is #data
            y: np.array with size N
        """
        self.w = np.dot(np.linalg.pinv(X),y)

    def pred(self,X):
        return np.dot(X,self.w)

class LogisticReg(object):

    def __init__(self,dim,fn=None):
        self.w = np.zeros(dim)
        if fn is None:
            self.fn_name = 'sigmoid'
        else:
            raise NotImplementedError("{} is not implemented".format(fn))

    def grad(self,X,y):
        if self.fn_name == 'sigmoid':
            N = X.shape[0]
            logis = sigmoid(-1* y * np.dot(X,self.w)) # (N,1)
            return np.dot(X.T, -1 * y * logis) / N

    def fit(self,X,y,eta,to_updates=INF,stochastic = False):
        cur_idx = 0
        N = X.shape[0]
        for t in range(to_updates):
            if stochastic:
                g = self.grad(X[cur_idx],y[cur_idx])
                cur_idx += 1
                if cur_idx == N:
                    cur_idx = 0
            else:
                g = self.grad(X,y)

            self.w += -1 * eta * g

            if np.linalg.norm(g)/self.w.shape[0] < EPS:
                break
           
    def pred(self,X):
        if self.fn_name == 'sigmoid':
            return np.rint(sigmoid(np.dot(X,self.w))) * 2 -1
