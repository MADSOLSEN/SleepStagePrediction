import numpy as np


from scipy.ndimage.morphology import binary_erosion, binary_dilation


def post_processing(prediction, merge_size, min_len):
    
    assert np.array_equal(prediction, prediction.astype(bool))

    # morphological closing - merge closely spaced events
    if merge_size > 0:
        prediction = binary_dilation(prediction, iterations=merge_size)
        prediction = binary_erosion(prediction, iterations=merge_size)

    # morphological opening - remove short events
    if min_len > 0:
        prediction = binary_erosion(prediction, iterations=min_len)
        prediction = binary_dilation(prediction, iterations=min_len)

    return prediction