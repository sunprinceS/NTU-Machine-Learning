import numpy as np

INF = int(1e9)

class PLA(object):
    def __init__(self,dim):
        self.w = np.zeros(dim)

    def fit(self,X,y,eta,rand_cycle,to_updates = INF):
        """
            X: np.array with size N * dim, N is #data
            y: np.array with size N
            rand_cycle: bool
            to_updates: int, max. #updates allowed
        """
        num_data = X.shape[0]
        vis_order = np.arange(num_data)
        if rand_cycle:
            np.random.shuffle(vis_order)

        num_update = 0

        while(num_update < to_updates):
            no_err = True

            for idx in vis_order:
                if(np.sign(np.dot(self.w.T,X[idx])) != y[idx]):
                    no_err = False
                    self.w += (eta * y[idx] * X[idx])
                    num_update += 1
                    if num_update == to_updates:
                        break;
            if no_err:
                break

        return num_update

class pocketPLA(PLA):
    def __init__(self,dim):
        super().__init__(dim)

    def fit(self,X,y,eta,to_updates):
        N = X.shape[0]

        w_opt = self.w
        err_ls = np.where(np.sign(np.dot(X,self.w)) != y)[0]
        min_err = len(err_ls)
        for t in range(to_updates):
            idx = np.random.choice(err_ls,1)[0]
            self.w += (eta * y[idx] * X[idx]) # we still need to update w but keep optimal

            pred = np.sign(np.dot(X,self.w))
            err_ls = np.where(pred != y)[0]
            err = len(err_ls)
            if err < min_err:
                w_opt = np.copy(self.w) ## Note: default is deep copy
                min_err = err

        self.w = w_opt
