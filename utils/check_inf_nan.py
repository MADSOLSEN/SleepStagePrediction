import numpy as np
from time import time

def check_inf_nan(X):
    #X = np.array(X)
    assert(np.sum(np.isinf(X)) == 0)
    assert(np.sum(np.isnan(X)) == 0)

