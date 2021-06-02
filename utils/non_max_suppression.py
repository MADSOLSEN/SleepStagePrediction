import numpy as np


def non_max_suppression(localizations, scores, overlap=0.5):
    """1D nms
        
        localizations: tensor of localizations
        (in array format [[start1, end1], [start2, end2], ...])

        score: softmax score of event 0 <= score <= 1.

        overlap: overlap between objects. 

    """
    x = localizations[:, 0]
    y = localizations[:, 1]

    areas = y - x
    order = np.argsort(scores, axis=0)[::-1]

    keep = []
    while order.size > 1:
        i = order[0]
        keep.append([x[i], y[i]])
        order = order[1:]

        # Original
        # xx = torch.clamp(x[order], min=x[i].item())
        # yy = torch.clamp(y[order], max=y[i].item())
        # intersection = torch.clamp(yy - xx, min=0)

        xx = np.clip(x[order], a_min=x[i].item(), a_max=None)
        yy = np.clip(y[order], a_max=y[i].item(), a_min=None)

        # xx = np.maximum(x[order], x[i].item())
        # yy = np.minimum(y[order], y[i].item())

        intersection = np.maximum(yy - xx, 0)
        intersection_over_union = intersection / (areas[i] + areas[order] - intersection)

        order = order[intersection_over_union <= overlap]

    keep.extend([[x[k], y[k]] for k in order])  # remaining element if order has size 1

    return keep
