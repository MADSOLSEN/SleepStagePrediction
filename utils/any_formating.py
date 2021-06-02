import numpy as np

def any_formating(events):
    """ Return binary array from [start, duration] array 

    output_size:    size if binary array to output.
    events format:  [[relative_start, relative_stop], [relative_start, relative_stop]]
    output: binary_array

    e.g. 
    semantic_formating(output_size=10, events=[[0.1, 0.2], [0.6, 0.8]])
    output: ([0, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    """

    if len(events) > 0:
        return np.ones((1,))
    else:
        return np.zeros((1,))
        






