import os
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import loguniform
# import cv2
import random

## This example using cubic splice is not the best approach to generate random curves. ## This
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
## inspired by: https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
##
def GenerateRandomCurves(X, sigma, max_drift_in_sec=2, fs=32):

    window = X.shape[0] // fs
    knot = window // max_drift_in_sec # number of zerocrossings, you could say.

    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    x_out = []

    for n in range(X.shape[-1]):
        cs = CubicSpline(xx[:, n], yy[:, n])
        x_out += [cs(x_range)]
    return np.array(x_out).transpose()


def DistortTimesteps(X, sigma):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    for n in range(X.shape[-1]):
        t_scale = [(X.shape[0] - 1) / tt_cum[-1, n]]
        tt_cum[:, n] = tt_cum[:, n] * t_scale
    return tt_cum

def TimeWarp(X, sigma=.1):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for n in range(X.shape[-1]):
        X_new[:, n] = np.interp(x_range, tt_new[:, n], X[:, n])
    return X_new


def MagWarp(X, sigma=0.1):
    return X * GenerateRandomCurves(X, sigma)


def shuffle_features(X):
    # assumes that the X has dimensions [T, F], where T is the time axis and F is the feature axis.
    NFeatures = X.shape[-1]
    feature_idx = list(range(NFeatures))
    random.shuffle(feature_idx)
    return X[:, feature_idx]


def Inverting(X, p=.5):
    invertingFactor = (np.random.binomial(n=1, p=p, size=(1, X.shape[1])) * 2) - 1
    myNoise = np.matmul(np.ones((X.shape[0], 1)), invertingFactor)
    return X * myNoise


def Jitter(X, sigma=0.1):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def Scaling(X, sigma=0.1):
    if len(X.shape) == 2:
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1])) # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    else:
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[2]))  # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0], X.shape[1], 1)), scalingFactor)
    return X * myNoise

#def insert_random_noise(X, sigma=0.1):

def image_translation(X, window=15, sigma=0.3):
    # sigma is the frequency the image can be translated.
    # e.g. 0.1 means signal can be tranlated from 1.5 Hz between 1.4-1.6 Hz

    fbinsize = 1/ (window) # fs / (window_length * 2)
    sigma_samples = int(sigma / fbinsize)

    translation_size = np.random.randint(low=-sigma_samples, high=sigma_samples, size=1)[0]

    if translation_size > 0:
        X_trans = np.zeros((X.shape)) + X.mean()
        X_trans[:, :-translation_size] = X[:, translation_size:]
    elif translation_size < 0:
        X_trans = np.zeros((X.shape)) + X.mean()
        X_trans[:, -translation_size:] = X[:, :translation_size]
    else:
        X_trans = X
    return X_trans


def freq_mask(spec, F=4, num_masks=2, replace_with_zero=False):
    cloned = np.array(spec)
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero != f_zero + f):
            mask_end = random.randrange(f_zero, f_zero + f)
            if (replace_with_zero):
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=15, num_masks=16, replace_with_zero=False):
    cloned = np.array(spec)
    len_spectro = cloned.shape[0]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero != t_zero + t):
            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero):
                cloned[t_zero:mask_end, :] = 0
            else:
                cloned[t_zero:mask_end, :] = cloned.mean()
    return cloned

def replace_with_noise(X, sample_prob=0.1, gauss_scale=0.2):
    X = np.array(X)
    if random.random() < sample_prob:
        X = np.random.normal(loc=X.mean(), scale=gauss_scale, size=X.shape)
    return X

def time_mask_USleep(X, sample_prob=0.1, low=.001, high=0.3, gauss_scale=0.01):
    # TODO - consider using mean of gauss per freq bin,
    X = np.array(X)
    k = 1
    if random.random() < sample_prob:
        seg_frag = loguniform.rvs(a=low, b=high)
        seg_len = int(seg_frag * X.shape[0])

        start_idx = np.random.randint(low=0, high=X.shape[0] - seg_len)
        gauss_seg = np.random.normal(loc=X.mean(), scale=gauss_scale, size=[seg_len] + list(X.shape[1:]))
        X[start_idx:start_idx + seg_len, :] = gauss_seg
    return X

def plot_stuff(X):

    t = np.linspace(0, X.shape[0] / 32, X.shape[0])
    transformations = {
        'Jitter': Jitter,
        'Scaling': Scaling,
        'MagWarp': MagWarp,
        'TimeWarp': TimeWarp,
    }

    gs = gridspec.GridSpec(5, 1)
    fig = plt.figure(figsize=(30, 10), dpi=200)
    fig.suptitle('ROC curves', fontsize=14)

    ax = fig.add_subplot(gs[0])
    ax.plot(t, X[:, 0], alpha=0.6, label='raw', markersize=6)
    ax.legend(prop={'size': 8}, loc=4)
    ax.grid(which='both')

    for n, trans in enumerate(transformations.keys()):  # augmentation
        ax = fig.add_subplot(gs[n + 1])
        X_ = transformations[trans](X)
        ax.plot(t, X_[:, 0], alpha=0.6, label=trans, markersize=6)

        ax.set_xlabel(r't', size=12)
        ax.set_ylabel(r'y', size=12)
        #ax.set_ylim([0.0, 1.0])
        #ax.set_xlim([0.0, 1.0])
        ax.legend(prop={'size': 8}, loc=4)
        ax.grid(which='both')

    label = '{}.png'.format('transformations')
    plt.savefig(os.path.join('D:\\Arc_study\\figures\\augmentation', label))