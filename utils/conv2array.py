from numpy cimport ndarray as ar
from time import time
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def toarr(xy):
    cdef int i, j, h=len(xy), w=len(xy[0])
    cdef ar[double,ndim=2] new = np.empty((h,w))
    for i in xrange(h):
        for j in xrange(w):
            new[i,j] = xy[i][j]
    return new

input = []
for n in range(1024):
    input += [np.zeros((3840, 7))]

t = time()
output = toarr(input)
print(time() - t)

t = time()
output = np.array(input)
print(time() - t)

