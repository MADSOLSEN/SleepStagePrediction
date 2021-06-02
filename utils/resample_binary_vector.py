import numpy as np

def resample_binary_vector(mask, fs, fs_desired):

    factor = (fs_desired / fs)
    if factor == 1:
        return np.array(mask)
    output_size = int(mask.shape[0]*factor)
    mask_out = np.zeros((output_size, mask.shape[1]))
    for c_n in range(mask.shape[1]):
        if factor < 1:
            mask_out[:, c_n] = mask[::int(1/factor), c_n]
        else:
            mask_out[:, c_n] = np.repeat(mask[:, c_n], int(factor))
    return mask_out