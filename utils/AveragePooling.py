import numpy as np
from numpy.lib.stride_tricks import as_strided
from skimage.measure import block_reduce


def pool1d(A, kernel_size, pool_mode='max'):
    '''
    1D Pooling

    Parameters:
        A: input 1D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding

    if pool_mode == 'max':
        return block_reduce(A, block_size=(kernel_size,), func=np.max)
    elif pool_mode == 'avg':
        return block_reduce(A, block_size=(kernel_size,), func=np.mean)


try_me = np.array(list(range(150)))
new_me = pool1d(try_me, kernel_size=30, pool_mode='avg')
k = 1

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)