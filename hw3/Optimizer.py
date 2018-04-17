import numpy as np

INF = int(1e6)
EPS = 1e-6

def grad_descent(X, y, grad_fn=None, batch_size = None, to_updates = INF ,tolerance = EPS):
    """
        X: np.array with size (N,dim), N is #data
        y: np.array with size (N,)
        cost_fn: error function
        batch_size: # of samples needed for update once
            if batch_size == None: iterate over all data
            if batch_size == 1: Stochastic GD
    """
    for t in range(to_updates):
        pass
