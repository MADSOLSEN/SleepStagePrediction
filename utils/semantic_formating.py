import numpy as np

def semantic_formating(output_size, events, sufficient_overlap=.25):
    """ Return binary array from [start, duration] array 

    output_size:    size if binary array to output.
    events format:  [[relative_start, relative_stop], [relative_start, relative_stop]]
    output: binary_array

    e.g. 
    semantic_formating(output_size=10, events=[[0.1, 0.2], [0.6, 0.8]])
    output: ([0, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    """

    output = np.zeros((output_size,))
    for event in events:
        # assert (event[0] < 1 and event[1] <= 1)

        if ((event[1] - event[0]) * output_size > sufficient_overlap) and (event[0] < 1 and event[1] <= 1):

            if event[0] * output_size % 1 <= (1 - sufficient_overlap):
                start_idx = int(event[0] * output_size)
            else:
                start_idx = round(event[0] * output_size)

            if event[1] * output_size % 1 >= (sufficient_overlap):
                stop_idx = int(event[1] * output_size) + 1
            else:
                stop_idx = round(event[1] * output_size)

            output[start_idx: stop_idx] = 1

    return output
