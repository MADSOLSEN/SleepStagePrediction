import numpy as np

def buffer_indexes(signal, window_length, overlap=0):

    signal_length = len(signal)
    W_STEP = round(window_length * (1 - overlap))
    N_WINDOWS = int(((signal_length - window_length) / (W_STEP)) + 1)

    windows = np.zeros((N_WINDOWS, window_length), dtype=int)

    for n in range(N_WINDOWS):
        windows[n, :] = range(n * W_STEP, window_length + n *W_STEP)

    return windows