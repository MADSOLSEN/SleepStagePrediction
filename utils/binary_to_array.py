import numpy as np


def binary_to_array(x):
    """ Return [start, duration] from binary array

    binary_to_array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    [[4, 8], [11, 13]]
    """
    tmp = np.array([0] + list(x) + [0])
    return np.where((tmp[1:] - tmp[:-1]) != 0)[0].reshape((-1, 2)).tolist()
