import numpy as np

INF = int(1e9)
EPS = 1e-6

def dec_stump(X,y):
    N = X.shape[0]
    min_err = INF
    theta_opt = -1
    s = 1

    for theta in X:
        cur_err = np.where(np.sign(X-theta + EPS) != y)[0].shape[0]
        if cur_err < min_err:
            min_err = cur_err
            theta_opt = theta

    for theta in X:
        cur_err = np.where(np.sign(X-theta + EPS) == y)[0].shape[0]
        if cur_err < min_err:
            min_err = cur_err
            theta_opt = theta
            s = -1

    E_i = min_err / N
    E_o = 0.5 + 0.3 * s * (np.abs(theta_opt)-1)

    return E_i,E_o

def multi_dec_stump(X,y):
    N = X.shape[0]
    dim = X.shape[1]
    X = X.T
    min_err = INF
    d_opt = 0
    theta_opt = INF
    s = 1

    for d in range(dim):
        for theta in X[d]:
            cur_err = np.where(np.sign(X[d]-theta + EPS) != y)[0].shape[0]
            if cur_err < min_err or (cur_err == min_err and np.random.randint(2,size=1)[0] == 0):
                min_err = cur_err
                theta_opt = theta
                d_opt = d

        for theta in X[d]:
            cur_err = np.where(np.sign(X[d]-theta + EPS) == y)[0].shape[0]
            if cur_err < min_err or (cur_err == min_err and np.random.randint(2,size=1)[0] == 0):
                min_err = cur_err
                theta_opt = theta
                d_opt = d
                s = -1
    E_i = min_err / N

    return E_i,theta_opt,d_opt,s
