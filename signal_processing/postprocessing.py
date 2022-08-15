import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage.morphology import binary_erosion, binary_dilation


def post_processing(prediction, merge_size, min_len, plot_flag=False):

    prediction_ori = prediction
    assert np.array_equal(prediction, prediction.astype(bool))

    # morphological closing - merge closely spaced events
    if merge_size > 0:
        prediction = binary_dilation(prediction, iterations=merge_size)
        prediction = binary_erosion(prediction, iterations=merge_size)

    # morphological opening - remove short events
    if min_len > 0:
        prediction = binary_erosion(prediction, iterations=min_len)
        prediction = binary_dilation(prediction, iterations=min_len)

    if plot_flag:

        fontsize = 6
        fig = plt.figure(figsize=(12, 1))
        gs = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0])

        ax.plot(prediction_ori, linewidth=.2)
        ax.plot(prediction, linewidth=.2)
        ax.set_title('merge size: {}. min len: {}'.format(merge_size, min_len), size=fontsize)
        ax.legend(['original', 'changed'], loc=4)

        plt.show()


    return prediction