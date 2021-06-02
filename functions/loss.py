import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def cross_entropy_loss(alpha=0.25, gamma=2.):
    """
    Binary form of focal loss.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def cross_entropy_loss_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """

        """
        if y_true.shape[0]:
            # t = np.array(y_true)
            p = np.array(y_pred)

            # print('true: -1: {}, 0: {}, 1: {}'.format(np.sum(t == -1), np.sum(t == 0), np.sum(t == 1)))
            print('pred: min: {}, max {}, mean {}'.format(np.min(p), np.max(p), np.mean(p)))
        """

        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        loss = (K.log(pt_1)) + \
               (K.log(1. - pt_0))

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(masked_loss)
        return masked_loss_mean

    return cross_entropy_loss_fixed

def weighted_loss(alpha=0.5, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras

    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def weighted_loss_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1]) / (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32)))
        pw = K.clip(pw, epsilon, 1. - epsilon)
        nw = 1 - pw

        loss = (K.pow(1. - pw, alpha) * K.log(pt_1)) + \
               (K.pow(1. - nw, alpha) * K.log(1. - pt_0))

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(masked_loss)
        return masked_loss_mean

    return weighted_loss_fixed


def effective_number_cross_entropy_loss(beta=0.99, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras

    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def effective_number_cross_entropy_loss_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1])
        nw = (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32))) - pw

        pos_weights = (1 - beta) / (1 - K.pow(beta, pw) + (1 - beta))
        neg_weights = (1 - beta) / (1 - K.pow(beta, nw) + (1 - beta))

        loss = (pos_weights / tf.reduce_sum(pos_weights) * K.log(pt_1)) + \
               (neg_weights / tf.reduce_sum(neg_weights) * K.log(1. - pt_0))


        #loss =  K.log(pt_1) + K.log(1. - pt_0)

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)

        masked_loss_mean = - tf.reduce_mean(masked_loss)

        # plot ():
        return masked_loss_mean

    return effective_number_cross_entropy_loss_fixed


def binary_focal_loss(alpha=0.25, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras

    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        loss = (K.pow(1. - pt_1, gamma) * K.log(pt_1)) + \
               (K.pow(pt_0, gamma) * K.log(1. - pt_0))

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(masked_loss)
        return masked_loss_mean

    return binary_focal_loss_fixed


def binary_focal_loss_weighted(alpha=0.5, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras

    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_weighted_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1]) / (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32)))
        pw = K.clip(pw, epsilon, 1. - epsilon)
        nw = 1 - pw

        loss = (K.pow(1. - pw, alpha) * K.pow(1. - pt_1, gamma) * K.log(pt_1)) + \
               (K.pow(1. - nw, alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(masked_loss)
        return masked_loss_mean

    return binary_focal_loss_weighted_fixed


def effective_number_binary_focal_loss(beta=0.99, gamma=2.):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        https://github.com/mkocabas/focal-loss-keras

    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def effective_number_binary_focal_loss_fixed(y_true, y_pred):
        """
        : param y_true: A tensor of the same shape as `y_pred`
        : param y_pred:  A tensor resulting from a sigmoid
        : return: Output tensor.
        """
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1])
        nw = (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32))) - pw

        loss = (((1-beta) / (1 - K.pow(beta, pw))) * K.pow(1. - pt_1, gamma) * K.log(pt_1)) + \
               (((1-beta) / (1 - K.pow(beta, nw))) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        masked_loss = tf.concat([tf.boolean_mask(loss, mask), tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(masked_loss)
        return masked_loss_mean

    return effective_number_binary_focal_loss_fixed

# Hard example mining:

def OHEM_cross_entropy_loss(rate, alpha=0.5, gamma=2., num_samples=32*64*4):
    """ Online hard negative mining. Includes only hard examples (either positive or negative) """
    def ohem_cross_entropy_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        loss = (K.log(pt_1)) + \
               (K.log(1. - pt_0))
        masked_loss = tf.boolean_mask(loss, mask)

        # sort
        loss_sorted = tf.sort(masked_loss, direction='ASCENDING')
        loss_hard = loss_sorted[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean
    return ohem_cross_entropy_loss


def OHEM_cross_entropy_weighted_loss(rate, alpha=0.5, gamma=2., num_samples=32*64*4):
    """ Online hard negative mining. Includes only hard examples (either positive or negative) """
    def ohem_cross_entropy_weighted_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1]) / (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32)))
        pw = K.clip(pw, epsilon, 1. - epsilon)
        nw = 1 - pw

        loss = (K.pow(1. - pw, alpha) * K.log(pt_1)) + \
               (K.pow(1. - nw, alpha) * K.log(1. - pt_0))
        masked_loss = tf.boolean_mask(loss, mask)

        # sort
        loss_sorted = tf.sort(masked_loss, direction='ASCENDING')
        loss_hard = loss_sorted[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean
    return ohem_cross_entropy_weighted_loss


def OHEM_cross_entropy_efficient_number_loss(rate, beta=0.99, gamma=2., num_samples=32 * 64 * 4):
    """ Online hard negative mining. Includes only hard examples (either positive or negative) """

    def ohem_cross_entropy_weighted_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1])
        nw = (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32))) - pw

        loss = (((1 - beta) / (1 - K.pow(beta, pw))) * K.log(pt_1)) + \
               (((1 - beta) / (1 - K.pow(beta, nw))) * K.log(1. - pt_0))
        masked_loss = tf.boolean_mask(loss, mask)

        # sort
        loss_sorted = tf.sort(masked_loss, direction='ASCENDING')
        loss_hard = loss_sorted[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean

    return ohem_cross_entropy_weighted_loss


def OHNM_cross_entropy_loss(rate, alpha=0.5, gamma=2., num_samples=32 * 64 * 4):
    """ Online hard negative mining. Includes all positives, only select hard negatives """

    def ohnm_cross_entropy_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        loss_pos = (K.log(pt_1))
        loss_neg = (K.log(1. - pt_0))
        masked_loss_pos = tf.boolean_mask(loss_pos, mask)
        masked_loss_neg = tf.boolean_mask(loss_neg, mask)

        # sort
        loss_sorted_neg = tf.sort(masked_loss_neg, direction='ASCENDING')
        loss_hard_neg = loss_sorted_neg[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard_neg, masked_loss_pos, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean

    return ohnm_cross_entropy_loss

def OHNM_cross_entropy_weighted_loss(rate, alpha=0.5, gamma=2., num_samples=32*64*4):
    """ Online hard negative mining. Includes only hard examples (either positive or negative) """
    def ohnm_cross_entropy_weighted_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1]) / (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32)))
        pw = K.clip(pw, epsilon, 1. - epsilon)
        nw = 1 - pw

        loss_pos = (K.pow(1. - pw, alpha) * K.log(pt_1))
        loss_neg = (K.pow(1. - nw, alpha) * K.log(1. - pt_0))
        masked_loss_pos = tf.boolean_mask(loss_pos, mask)
        masked_loss_neg = tf.boolean_mask(loss_neg, mask)

        # sort
        loss_sorted_neg = tf.sort(masked_loss_neg, direction='ASCENDING')
        loss_hard_neg = loss_sorted_neg[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard_neg, masked_loss_pos, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean
    return ohnm_cross_entropy_weighted_loss


def OHNM_cross_entropy_efficient_number_loss(rate, beta=0.99, gamma=2., num_samples=32 * 64 * 4):
    """ Online hard negative mining. Includes only hard examples (either positive or negative) """

    def ohnm_cross_entropy_weighted_loss(y_true, y_pred):
        epsilon = K.epsilon()
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        # mask
        mask = K.greater_equal(y_true, -0.5)

        # weights
        masked_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        pw = tf.reduce_sum(masked_y_true, axis=[0, 1])
        nw = (tf.reduce_sum(tf.cast(mask[:, :, 0], tf.float32))) - pw

        loss_pos = (((1 - beta) / (1 - K.pow(beta, pw))) * K.log(pt_1))
        loss_neg = (((1 - beta) / (1 - K.pow(beta, nw))) * K.log(1. - pt_0))
        masked_loss_pos = tf.boolean_mask(loss_pos, mask)
        masked_loss_neg = tf.boolean_mask(loss_neg, mask)

        # sort
        loss_sorted_neg = tf.sort(masked_loss_neg, direction='ASCENDING')
        loss_hard_neg = loss_sorted_neg[:int(num_samples * rate)]

        loss_zeropadded = tf.concat([loss_hard_neg, masked_loss_pos, tf.constant([epsilon])], axis=0)
        masked_loss_mean = - tf.reduce_mean(loss_zeropadded)
        return masked_loss_mean

    return ohnm_cross_entropy_weighted_loss

def categorical_crossentropy_loss(gamma=2., alpha=.25):
    def categorical_crossentropy(output, target):
        # scale preds so that the class probas of each sample sum to 1
        output /= K.sum(output, axis=-1, keepdims=True)
        output = K.clip(output, K.epsilon(), 1. - K.epsilon())
        cross_entropy = target * K.log(output)
        return - K.mean(cross_entropy)

    return categorical_crossentropy

def categorical_crossentropy_weighted_loss(gamma=2., alpha=.5):
    def categorical_crossentropy_weighted(y_true, y_pred):
        # scale predictions
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        # weights
        pw = tf.reduce_mean(y_true, [0, 1])  # event proportion from batch
        pw = K.clip(pw, K.epsilon(), 1. - K.epsilon())

        # Calculate Cross Entropy
        cross_entropy = y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = K.pow(1. - pw, alpha) * cross_entropy

        # Sum the losses in mini_batch
        return - K.mean(loss)

    return categorical_crossentropy_weighted


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.mean(loss)

    return categorical_focal_loss_fixed

def categorical_focal_loss_weighted(gamma=2., alpha=.5):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # weights
        pw = tf.reduce_mean(y_true, [0, 1])  # event proportion from batch
        pw = K.clip(pw, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = K.pow(1. - pw, alpha) * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.mean(loss)

    return categorical_focal_loss_fixed

def DICED_LOSS():
    def diced_loss(y_true, y_pred):

        epsilon = K.epsilon()

        nominator = tf.math.multiply(y_true, y_pred)
        denominator = y_true + y_pred

        # remove mask
        mask = K.greater_equal(y_true, -0.5)
        nominator_masked = tf.concat([tf.boolean_mask(nominator, mask), tf.constant([epsilon])], axis=0)
        denominator_masked = tf.concat([tf.boolean_mask(denominator, mask), tf.constant([epsilon])], axis=0)

        loss = 1 - (tf.math.multiply(1/2, tf.reduce_sum(nominator_masked) / (tf.reduce_sum(denominator_masked) + K.epsilon())))

        # non-masked
        # nominator = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
        # denominator = tf.reduce_sum(y_true + y_pred)
        # loss = 1 - (tf.math.multiply(1/2, nominator / (denominator + K.epsilon())))

        return loss
    return diced_loss