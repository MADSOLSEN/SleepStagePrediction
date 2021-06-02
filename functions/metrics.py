import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from operator import truediv
import tensorflow as tf
import sys

from utils import binary_to_array, jaccard_overlap, semantic_formating

def calculate_AUCRC(lower_bound=0.01, upper_bound=0.99, num_samples=100):
    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=num_samples)

    def AUPRC(y_true, y_pred):
        recall_prev, AUPRC_ = 0, 0
        for n, threshold in enumerate(thresholds):

            y_pred_pos = K.round(K.clip((y_pred + (1 - threshold) - .5), K.epsilon(), 1. - K.epsilon()))
            y_pred_neg = 1 - y_pred_pos

            y_pos = y_true
            y_neg = 1 - y_pos
            tp = K.sum(y_pos * y_pred_pos)

            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)

            precision = tp / (tp + fp + K.epsilon())
            recall = tp / (tp + fn + K.epsilon())

            if n > 0:
                AUPRC_ += precision * (recall_prev - recall)
            recall_prev = recall
        return AUPRC_
    return AUPRC


def calculate_AUROC(lower_bound=0.01, upper_bound=0.99, num_samples=100):
    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=num_samples)

    def AUROC(y_true, y_pred):
        FPR_prev, AUROC_ = 0, 0
        for n, threshold in enumerate(thresholds):

            y_pred_pos = K.round(K.clip((y_pred + (1 - threshold) - .5), K.epsilon(), 1. - K.epsilon()))
            y_pred_neg = 1 - y_pred_pos

            y_pos = y_true
            y_neg = 1 - y_pos
            tp = K.sum(y_pos * y_pred_pos)
            tn = K.sum(y_neg * y_pred_neg)

            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)

            specificity = tn / (tn + fp + K.epsilon())
            FPR = 1 - specificity
            recall = tp / (tp + fn + K.epsilon())

            if n > 0:
                AUROC_ += recall * (FPR_prev - FPR)
            FPR_prev = FPR
        return AUROC_
    return AUROC


def optimized_f1(lower_bound=.2, upper_bound=.8, num_samples=100, beta=1):

    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=num_samples)

    def f1_(y_true, y_pred):
        f1_all = 0
        for threshold in thresholds:
            y_pred_pos = K.round(K.clip((y_pred + (1 - threshold) - .5), K.epsilon(), 1. - K.epsilon()))
            y_pred_neg = 1 - y_pred_pos

            y_pos = y_true
            y_neg = 1 - y_pos
            tp = K.sum(y_pos * y_pred_pos)

            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)

            precision = tp / (tp + fp + K.epsilon())
            recall = tp / (tp + fn + K.epsilon())

            nominator = (1 + beta ** 2) * (precision * recall)
            denominator = (beta ** 2 * precision) + recall
            f1 = nominator / (denominator + K.epsilon())
            f1_all += f1

        return f1_all / num_samples
    return f1_


def optimized_f12(num_classes, lower_bound=.2, upper_bound=.8, num_samples=100, beta=1):

    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=num_samples)

    def f1_2(y_true, y_pred):
        f1_all = 0
        for threshold in thresholds:
            y_pred_pos = K.round(K.clip((y_pred + (1 - threshold) - .5), K.epsilon(), 1. - K.epsilon()))
            y_pred_neg = 1 - y_pred_pos

            y_pos = y_true
            y_neg = 1 - y_pos

            f1 = 0
            for c in range(num_classes):
                tp = K.sum(y_pos[:, :, c] * y_pred_pos[:, :, c])
                fp = K.sum(y_neg[:, :, c] * y_pred_pos[:, :, c])
                fn = K.sum(y_pos[:, :, c] * y_pred_neg[:, :, c])

                precision = tp / (tp + fp + K.epsilon())
                recall = tp / (tp + fn + K.epsilon())

                nominator = (1 + beta ** 2) * (precision * recall)
                denominator = (beta ** 2 * precision) + recall
                f1 += nominator / (denominator + K.epsilon())
            f1_all += f1 / num_classes
        return f1_all / num_samples
    return f1_2


def binary_accuracy():
    def binary_accuracy(y_true, y_pred):

        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_pos = y_true
        y_neg = 1 - y_pos

        mask = K.greater_equal(y_true, -0.5)

        tp = K.sum(tf.boolean_mask(y_pos * y_pred_pos, mask))
        tn = K.sum(tf.boolean_mask(y_neg * y_pred_neg, mask))
        fp = K.sum(tf.boolean_mask(y_neg * y_pred_pos, mask))
        fn = K.sum(tf.boolean_mask(y_pos * y_pred_neg, mask))

        numerator = (tp + tn)
        denominator = (tp + tn + fp + fn)

        return numerator / (denominator + K.epsilon())

    return binary_accuracy


def cohens_kappa(num_classes):
    # TODO - this is wrong - in extreme cases we get negative and values above 1.
    def cohens_kappa(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_pos = y_true
        y_neg = 1 - y_pos

        mask = K.greater_equal(y_true, -0.5)

        kappa = 0
        for c in range(num_classes):
            tp = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            tn = K.sum(tf.boolean_mask(y_neg[:, :, c] * y_pred_neg[:, :, c], mask[:, :, c]))
            fp = K.sum(tf.boolean_mask(y_neg[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            fn = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_neg[:, :, c], mask[:, :, c]))

            N = (tp + tn + fp + fn)
            po = (tp + tn) / (N + K.epsilon())
            pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (N ** 2 + K.epsilon())
            kappa += (po - pe) / (1 - pe + K.epsilon())
        return kappa / num_classes
    return cohens_kappa


def cohens_kappa1(num_classes, lower_bound=.2, upper_bound=.8, num_samples=100, beta=1):
    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=num_samples)
    def cohens_kappa2(y_true, y_pred):
        kappa_all = 0
        for threshold in thresholds:
            y_pred_pos = K.round(K.clip((y_pred + (1 - threshold) - .5), K.epsilon(), 1. - K.epsilon()))
            y_pred_neg = 1 - y_pred_pos

            y_pos = y_true
            y_neg = 1 - y_pos

            kappa = 0
            for c in range(num_classes):
                tp = K.sum(y_pos[:, :, c] * y_pred_pos[:, :, c])
                tn = K.sum(y_neg[:, :, c] * y_pred_neg[:, :, c])

                fp = K.sum(y_neg[:, :, c] * y_pred_pos[:, :, c])
                fn = K.sum(y_pos[:, :, c] * y_pred_neg[:, :, c])

                N = (tp + tn + fp + fn)
                po = (tp + tn) / N
                pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / N ** 2
                kappa += (po - pe) / (1 - pe + K.epsilon())
            kappa_all += kappa / num_classes
        return kappa_all / num_samples
    return cohens_kappa2

def balanced_accuracy_new(num_classes):
    def balanced_accuracy_new(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_pos = y_true
        y_neg = 1 - y_pos

        mask = K.greater_equal(y_true, -0.5)

        balanced_accuracy = 0
        for c in range(num_classes):
            tp = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            fp = K.sum(tf.boolean_mask(y_neg[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            fn = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_neg[:, :, c], mask[:, :, c]))

            balanced_accuracy += (tp / tf.maximum((tp + fn + K.epsilon()), (tp + fp + K.epsilon())))
        return balanced_accuracy / num_classes
    return balanced_accuracy_new

def balanced_accuracy(num_classes):
    # TODO - this is wrong - in extreme cases we get negative and values above 1.
    def balanced_accuracy(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_pos = y_true
        y_neg = 1 - y_pos

        mask = K.greater_equal(y_true, -0.5)

        balanced_accuracy = 0
        for c in range(num_classes):
            tp = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            tn = K.sum(tf.boolean_mask(y_neg[:, :, c] * y_pred_neg[:, :, c], mask[:, :, c]))
            fp = K.sum(tf.boolean_mask(y_neg[:, :, c] * y_pred_pos[:, :, c], mask[:, :, c]))
            fn = K.sum(tf.boolean_mask(y_pos[:, :, c] * y_pred_neg[:, :, c], mask[:, :, c]))

            balanced_accuracy += (tp / (tp + fn + K.epsilon()) + tn / (tn + fp + K.epsilon())) / 2
        return balanced_accuracy / num_classes
    return balanced_accuracy


def MCC(y_true, y_pred):
    # REMEMBER: Only true when negative background label is 0.

    y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
    y_pred_neg = 1 - y_pred_pos

    y_pos = y_true
    y_neg = 1 - y_pos

    mask = K.greater_equal(y_true, -0.5)

    tp = K.sum(tf.boolean_mask(y_pos * y_pred_pos, mask))
    tn = K.sum(tf.boolean_mask(y_neg * y_pred_neg, mask))
    fp = K.sum(tf.boolean_mask(y_neg * y_pred_pos, mask))
    fn = K.sum(tf.boolean_mask(y_pos * y_pred_neg, mask))

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision_mean():
    def precision(y_true, y_pred):

        # scale and clip the prediction value to prevent NaN's and Inf's
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))

        y_true_pos = y_true
        y_true_neg = 1 - y_true_pos

        mask = K.greater_equal(y_true, -0.5)

        tp = K.sum(tf.boolean_mask(y_pred_pos * y_true_pos, mask))
        fp = K.sum(tf.boolean_mask(y_pred_pos * y_true_neg, mask))

        return (tp) / (tp + fp + K.epsilon())
    return precision


def specificity_mean():
    def specificity(y_true, y_pred):

        # scale and clip the prediction value to prevent NaN's and Inf's
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_true_pos = y_true
        y_true_neg = 1 - y_true_pos

        mask = K.greater_equal(y_true, -0.5)

        tn = K.sum(tf.boolean_mask(y_pred_neg * y_true_neg), mask)
        fp = K.sum(tf.boolean_mask(y_pred_pos * y_true_neg), mask)

        return (tn) / (tn + fp + K.epsilon())
    return specificity


def recall_mean():
    def recall(y_true, y_pred):

        # scale and clip the prediction value to prevent NaN's and Inf's
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_true_pos = y_true

        mask = K.greater_equal(y_true, -0.5)

        tp = K.sum(tf.boolean_mask(y_pred_pos * y_true_pos, mask))
        fn = K.sum(tf.boolean_mask(y_pred_neg * y_true_pos, mask))

        return (tp) / (tp + fn + K.epsilon())
    return recall


def f1_mean_metric(beta=1):
    def f1_mean(y_true, y_pred):

        # scale and clip the prediction value to prevent NaN's and Inf's
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_true_pos = y_true
        y_true_neg = 1 - y_true_pos

        mask = K.greater_equal(y_true, -0.5)

        tp = K.sum(tf.boolean_mask(y_pred_pos * y_true_pos, mask))
        fn = K.sum(tf.boolean_mask(y_pred_neg * y_true_pos, mask))
        fp = K.sum(tf.boolean_mask(y_pred_pos * y_true_neg, mask))

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        nominator = (1 + beta ** 2) * (precision * recall)
        denominator = (beta ** 2 * precision) + recall

        return nominator / (denominator + K.epsilon())
    return f1_mean

def f1_by_class_metric(beta=1, class_idx=0):
    def f1_by_class(y_true, y_pred):

        # select class
        y_true = y_true[:, :, class_idx]
        y_pred = y_pred[:, :, class_idx]

        # scale and clip the prediction value to prevent NaN's and Inf's
        y_pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1. - K.epsilon()))
        y_pred_neg = 1 - y_pred_pos

        y_true_pos = y_true
        y_true_neg = 1 - y_true_pos

        mask = K.greater_equal(y_true, -0.5)

        tp = K.sum(tf.boolean_mask(y_pred_pos * y_true_pos, mask))
        fn = K.sum(tf.boolean_mask(y_pred_neg * y_true_pos, mask))
        fp = K.sum(tf.boolean_mask(y_pred_pos * y_true_neg, mask))

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        nominator = (1 + beta ** 2) * (precision * recall)
        denominator = (beta ** 2 * precision) + recall

        return nominator / (denominator + K.epsilon())
    return f1_by_class


def get_false_positive_events(target_loc, output_loc, min_iou=0.5):

    if len(target_loc) == 0 and len(output_loc) > 0:
        false_positives = output_loc
    elif len(output_loc) == 0:
        false_positives = []
    else:
        iou = jaccard_overlap(np.array(target_loc),
                              np.array(output_loc))
        max_iou = iou.max(0)
        false_positives = [o_l for m_i, o_l in zip(max_iou, output_loc) if m_i < min_iou]
    return false_positives


def get_false_negative_events(target_loc, output_loc, min_iou=0.5):
    if len(output_loc) == 0 and len(target_loc) > 0:
        false_negatives = target_loc
    elif len(target_loc) == 0:
        false_negatives = []
    else:
        iou = jaccard_overlap(np.array(target_loc),
                              np.array(output_loc))
        max_iou = iou.max(1)
        false_negatives = [t_l for m_i, t_l in zip(max_iou, target_loc) if m_i < min_iou]
    return false_negatives


def get_true_positive_events(target_loc, output_loc, min_iou=0.5):
    if len(target_loc) == 0 or len(output_loc) == 0:
        true_positives = []
    else:
        iou = jaccard_overlap(np.array(target_loc),
                              np.array(output_loc))
        max_iou = iou.max(1)
        true_positives =  [t_l for m_i, t_l in zip(max_iou, target_loc) if m_i >= min_iou]
    return true_positives


def precision_function():
    """takes 2 events scorings
    (in binary array format [0, 0, 1, 1, 1, 0, 0, 1])
    and outputs precision

    Parameters
    ----------
    min_iou : float
        minimum intersection-over-union with a true event to be considered a true positive

    Explanation
    -----------
    Precision tells the proportion of your predicted positives are correct.
    If you don't predict any outputs we have 100% correct.
    """
    
    def calculate_precision(target_loc, output_loc, min_iou=0.5):
        # compute precision
        # target_loc = binary_to_array(target)
        # output_loc = binary_to_array(output)
        if len(target_loc) == 0 and len(output_loc) > 0:
            true_positive = 0
            false_positive = len(output_loc)
            precision = true_positive / (true_positive + false_positive)
        elif len(output_loc) == 0:
            precision = 1
            true_positive = 0
            false_positive = 0
        else:
            iou = jaccard_overlap(np.array(target_loc),
                                  np.array(output_loc))
            max_iou = iou.max(0)
            true_positive = (max_iou >= min_iou).sum()
            false_positive = len(output_loc) - true_positive
            precision = true_positive / (true_positive + false_positive)
        return precision
    
    return calculate_precision


def recall_function():
    """takes 2 events scorings
    (in binary array format [0, 0, 1, 1, 1, 0, 0, 1])
    and outputs recall

    Parameters
    ----------
    min_iou : float
        minimum intersection-over-union with a true event to be considered a true positive
    """
    
    def calculate_recall(target_loc, output_loc, min_iou=0.5):
        # target_loc = binary_to_array(target)
        # output_loc = binary_to_array(output)
        if len(output_loc) == 0 and len(target_loc) > 0:
            true_positive = 0
            false_negative = len(target_loc)
            recall = true_positive / (true_positive + false_negative)
        elif len(target_loc) == 0:
            recall = 1
            false_negative = 0
            true_positive = 0
        else:
            iou = jaccard_overlap(np.array(target_loc),
                                  np.array(output_loc))
            max_iou = iou.max(1)
            true_positive = (max_iou >= min_iou).sum()
            false_negative = len(target_loc) - true_positive
            recall = true_positive / (true_positive + false_negative)
        return recall
    return calculate_recall


def f1_function():
    """takes 2 events scorings
            (in binary array format [0, 0, 1, 1, 1, 0, 0, 1])
            and outputs F1

            Parameters
            ----------
            min_iou : float
                minimum intersection-over-union with a true event to be considered a true positive
            """

    def calculate_f1(target_loc, output_loc, min_iou=0.5):
        
        calculate_pre = precision_function()
        calculate_rec = recall_function()

        precision = calculate_pre(target_loc, output_loc, min_iou=min_iou)
        recall = calculate_rec(target_loc, output_loc, min_iou=min_iou)

        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)
        return F1
    return calculate_f1

def accuracy_semantic_input_function():

    def calculate_accuracy_semantic_input(target, output):
        target = target + [1]
        output = output + [0]
        tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=output).ravel()
        fn -= 1
        accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-7)
        return accuracy
    return calculate_accuracy_semantic_input


def recall_semantic_input_function():
    def calculate_recall_semantic_input(target, output):
        target = target + [1]
        output = output + [0]
        tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=output).ravel()
        fn -= 1
        recall = (tp) / (tp + fn + 1e-7)
        return recall

    return calculate_recall_semantic_input


def precision_semantic_input_function():
    def calculate_precision_semantic_input(target, output):
        target = target + [1]
        output = output + [0]
        tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=output).ravel()
        fn -= 1
        precision = (tp) / (fp + tp + 1e-7)
        return precision

    return calculate_precision_semantic_input


def specificity_semantic_input_function():
    def calculate_specificity_semantic_input(target, output):
        target = target + [1]
        output = output + [0]
        tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=output).ravel()
        fn -= 1
        specificity = (tn) / (tn + fp)
        return specificity

    return calculate_specificity_semantic_input
