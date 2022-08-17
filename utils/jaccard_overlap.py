import numpy as np
import sys

def jaccard_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)
    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    (in array format [[start1, end1], [start2, end2], ...])
    """
    assert localizations_a.shape[1] == 2 and localizations_a.shape[1] == 2 # ensure correct formatting

    A = localizations_a.shape[0]
    B = localizations_b.shape[0]
    # intersection
    if (B == 0) or (A == 0):
        return np.zeros((A, B))
    max_start = np.maximum(np.repeat(np.expand_dims(localizations_a[:, 0], axis=1), B, axis=1),
                           np.repeat(np.expand_dims(localizations_b[:, 0], axis=0), A, axis=0))
    min_stop = np.minimum(np.repeat(np.expand_dims(localizations_a[:, 1], axis=1), B, axis=1),
                          np.repeat(np.expand_dims(localizations_b[:, 1], axis=0), A, axis=0))

    intersection = np.maximum((min_stop - max_start), 0)
    length_a = np.repeat(np.expand_dims(localizations_a[:, 1] - localizations_a[:, 0], axis=1), B, axis=1)
    length_b = np.repeat(np.expand_dims(localizations_b[:, 1] - localizations_b[:, 0], axis=0), A, axis=0)

    # IoU: intersection over union
    overlaps = intersection / (length_a + length_b - intersection + sys.float_info.epsilon)
    return overlaps


def intersection_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B
    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    (in array format [[start1, end1], [start2, end2], ...])
    """
    assert localizations_a.shape[1] == 2 and localizations_a.shape[1] == 2  # ensure correct formatting

    A = localizations_a.shape[0]
    B = localizations_b.shape[0]

    if (B == 0) or (A == 0):
        return np.zeros((A, B))

    # intersection
    max_start = np.maximum(np.repeat(np.expand_dims(localizations_a[:, 0], axis=1), B, axis=1),
                           np.repeat(np.expand_dims(localizations_b[:, 0], axis=0), A, axis=0))
    min_stop = np.minimum(np.repeat(np.expand_dims(localizations_a[:, 1], axis=1), B, axis=1),
                          np.repeat(np.expand_dims(localizations_b[:, 1], axis=0), A, axis=0))

    intersection = np.maximum((min_stop - max_start), 0)
    return intersection

def relative_start_by_intersection(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B
    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    (in array format [[start1, end1], [start2, end2], ...])
    """
    assert localizations_a.shape[1] == 2 and localizations_a.shape[1] == 2  # ensure correct formatting

    A = localizations_a.shape[0]
    B = localizations_b.shape[0]

    if (B == 0) or (A == 0):
        return np.zeros((A, B))

    # intersection
    max_start = np.maximum(np.repeat(np.expand_dims(localizations_a[:, 0], axis=1), B, axis=1),
                           np.repeat(np.expand_dims(localizations_b[:, 0], axis=0), A, axis=0))
    min_stop = np.minimum(np.repeat(np.expand_dims(localizations_a[:, 1], axis=1), B, axis=1),
                          np.repeat(np.expand_dims(localizations_b[:, 1], axis=0), A, axis=0))
    intersection = np.maximum((min_stop - max_start), 0)

    # relative start
    rel_start = np.repeat(np.expand_dims(localizations_b[:, 0], axis=0), A, axis=0) - \
                np.repeat(np.expand_dims(localizations_a[:, 0], axis=1), B, axis=1)
    rel_stop = np.repeat(np.expand_dims(localizations_b[:, 1], axis=0), A, axis=0) - \
               np.repeat(np.expand_dims(localizations_a[:, 1], axis=1), B, axis=1)

    # select only intersection
    rel_start[intersection <= 0] = 1e6

    return rel_start


#sdb = [[50, 65], [80, 100]]
#light = [[0, 30],[30, 60],[60, 90]]
#out = relative_start_by_intersection(np.array(sdb), np.array(light))
#print(out)