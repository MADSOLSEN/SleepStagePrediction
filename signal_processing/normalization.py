import numpy as np
from scipy.stats import zscore
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
import sys


def adaptive_soft(x, fs, median_window, std_window):

    def normalize(x, fs, median_window, std_window):
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        med_window = (fs * median_window + 1) if (fs * median_window) % 2 == 0 else (fs * median_window)
        std_window = (fs * std_window + 1) if (fs * std_window) % 2 == 0 else (fs * std_window)
        x_med = np.ones((x.shape)) * np.median(x_)
        x_std = np.ones((x.shape)) * np.std(x_)

        x_pd = pd.Series(x_)
        med_ = x_pd.rolling(med_window).median()
        x_med[int(med_window / 2):-int(med_window / 2)] = med_[med_window - 1:]
        x_med[np.isnan(x_med)] = 0  # remove nan

        x_ = (x_ - x_med)

        x_pd = pd.Series(x_)
        std_ = x_pd.rolling(std_window).std()
        x_std[int(std_window / 2):-int(std_window / 2)] = std_[std_window - 1:]
        x_std[np.isnan(x_std)] = 0  # remove nan

        return x_ / (x_std + 1e-10)

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x, fs, median_window, std_window)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n], fs, median_window, std_window)
    #plt.plot(x)
    #plt.show()
    #plt.plot(x_out)
    #plt.show()
    return x_norm

def zscore_log(x, fs, min_value, max_value):

    if x.max() == x.min():
        return x
    x_clip = np.clip(x, a_min=min_value, a_max=max_value)
    x_log = np.log(x_clip + 1)
    x_norm = (x_log - np.nanmean(x_log)) / np.nanstd(x_log)
    return x_norm

def downsample(x, fs, fs_out, plot=False):
    if int(fs) is not int(fs_out):
        x_out = signal.resample_poly(x, up=1, down=fs // fs_out)

        if plot:
            f, t, Sxx = signal.spectrogram(x, fs)
            plt.pcolormesh(t, f, np.log2(Sxx), shading='gouraud')
            plt.show()

            f, t, Sxx = signal.spectrogram(x_out, fs_out)
            plt.pcolormesh(t, f, np.log2(Sxx), shading='gouraud')
            plt.show()
        return (x_out, fs_out)
        #step = int(fs / fs_out)
        #return x[::step]
    else:
        return (x, fs_out)


def zscore_norm(x, fs=1, axis=0):

    assert(len(x.shape) < 3)

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm = (x - x.mean()) / (x.std() + sys.float_info.epsilon)
    elif len(x.shape) == 2:
        for idx in range(x.shape[1]):
            x_norm[:, idx] = (x[:, idx] - x[:, idx].mean()) / (x[:, idx].std() + sys.float_info.epsilon)

    return x_norm

def divide(x, fs, val):

    x_out = x / val
    return x_out


def min_max(x):

    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def soft(x, fs, perc_low, perc_high):

    if len(x.shape) == 3:
        x_soft = np.zeros((x.shape))
        for channels in range(x.shape[-1]):
            q_low = np.percentile(x[..., channels], perc_low)
            q_high = np.percentile(x[..., channels], perc_high)
            if q_low != q_high:
                x_soft[..., channels] = (x[..., channels] - q_low)/(q_high-q_low)
    return x_soft


def median(x, fs, window_size):

    window = (fs * window_size + 1) if (fs * window_size) % 2 == 0 else (fs * window_size)

    reduce_dims = False
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=-1)
        reduce_dims = True
    x_norm = np.zeros((x.shape))

    for idx in range(x.shape[-1]):

        x_med = np.ones((x.shape[0])) * np.median(x[:, idx])

        x_pd = pd.Series(x[:, idx])
        med_ = x_pd.rolling(window).median()
        x_med[int(window / 2):-int(window / 2)] = med_[window - 1:]
        x_med[:int(window / 2)] = med_[window - 1]
        x_med[-int(window / 2):] = med_[-1:]

        x_med[np.isnan(x_med)] = 0  # remove nan

        x_norm[:, idx] = x[:, idx] - x_med

    if reduce_dims:
        x_norm = x_norm[:, 0]
    return x_norm

    #plt.plot(x[int(1316992 / 2):int(1316992 / 2 + 60 * 32)], linewidth=0.5)

def median_filter(x, fs, window_size):

    window = (fs * window_size + 1) if (fs * window_size) % 2 == 0 else (fs * window_size)
    x_med = np.ones((x.shape)) * np.median(x)

    x_pd = pd.Series(x)
    med_ = x_pd.rolling(window).median()
    x_med[int(window / 2):-int(window / 2)] = med_[window - 1:]
    x_med[np.isnan(x_med)] = 0  # remove nan

    return x_med


def complete_zscore(x):
    x_norm = zscore(x, axis=None)
    return x_norm


def zscore_axis(x):
    x_norm = np.zeros((x.shape))
    for n in range(x.shape[1]):
        if np.min(x[:, n]) == np.max(x[:, n]):
            x_norm[:, n] = x[:, n]
        else:
            x_norm[:, n] = (x[:, n] - np.nanmean(x[:, n])) / np.nanstd(x[:, n])
    return x_norm


def diff(x, fs):
    x_out = np.zeros((x.shape))
    x_out[1:] = np.diff(x, axis=-1)
    return x_out


def clip(x, lower, upper):
    x[x > upper] = upper
    x[x < lower] = lower
    return x


def no_normalization(x):
    return x


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(x, fs, lowcut, highcut, order=5, plot=False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, x)

    return y

def butter_highpass_filter(x, fs, highcut, order=10, plot=False):
    sos = signal.butter(order, highcut, 'hp', fs=fs, output='sos')
    y = signal.sosfilt(sos, x)

    k = 1
    if plot:
        plt.plot(x[2 * 3600 * fs: (2 * 3600 + 10) * fs])
        plt.plot(y[2 * 3600 * fs: (2 * 3600 + 10) * fs])
        plt.show()

    return y

def clip_by_iqr(x, fs, threshold=20):

    x[x > threshold] = threshold
    x[x < -threshold] = - threshold

    return x

def iqr_normalization(x, fs, iqr_upper=0.75, iqr_lower=0.25, plot=False):

    def normalize(x):
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        x_iqr_up = np.quantile(x_, iqr_upper)
        x_iqr_lo = np.quantile(x_, iqr_lower)

        return (x_ - x_iqr_lo) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon) * 2 - 1

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n])

    if plot:
        plt.plot(x[3600*32*6:3600*32*6+60*32])
        plt.plot(x_norm[3600*32*6:3600*32*6+60*32])
        plt.show()

        k = 1
    return x_norm


def iqr_normalization_adaptive(x, fs, median_window, iqr_window):
    def normalize(x, fs, median_window, iqr_window, iqr_upper=0.75, iqr_lower=0.25):

        # add noise
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        # fix window parameters to odd number
        med_window = (fs * median_window + 1) if (fs * median_window) % 2 == 0 else (fs * median_window)
        iqr_window = (fs * iqr_window + 1) if (fs * iqr_window) % 2 == 0 else (fs * iqr_window)

        # preallocation
        x_med = np.ones((x.shape)) * np.median(x_)
        x_iqr_up = np.ones((x.shape)) * np.quantile(x_, iqr_upper)
        x_iqr_lo = np.ones((x.shape)) * np.quantile(x_, iqr_lower)

        # find rolling median
        x_pd = pd.Series(x_)
        med_ = x_pd.rolling(med_window).median()
        x_med[int(med_window / 2):-int(med_window / 2)] = med_[med_window - 1:]
        x_med[np.isnan(x_med)] = 0  # remove nan

        # find rolling quantiles
        x_iqr_upper = x_pd.rolling(iqr_window).quantile(iqr_upper)
        x_iqr_lower = x_pd.rolling(iqr_window).quantile(iqr_lower)

        # border padding
        x_iqr_up[int(iqr_window / 2):-int(iqr_window / 2)] = x_iqr_upper[iqr_window - 1:]
        x_iqr_lo[int(iqr_window / 2):-int(iqr_window / 2)] = x_iqr_lower[iqr_window - 1:]

        # remove nan
        x_iqr_up[np.isnan(x_iqr_up)] = 0
        x_iqr_lo[np.isnan(x_iqr_lo)] = 0

        # return normalize
        return (x_ - x_iqr_lo) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon) * 2 - 1

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x, fs, median_window, iqr_window)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n], fs, median_window, iqr_window)
    return x_norm

def change_PPG_direction(x, fs, plot=False):
    import os

    #x_above_zero = x[(x > 0) & (np.abs(np.gradient(x)) > 0)]
    #x_below_zero = x[(x < 0) & (np.abs(np.gradient(x)) > 0)]

    x_above_zero = x[(x > 0) & (np.abs(np.gradient(x)) > 0)]
    x_below_zero = x[(x < 0) & (np.abs(np.gradient(x)) > 0)]

    x_diff_above = np.median(np.abs(np.diff(x_above_zero)))
    x_diff_below = np.median(np.abs(np.diff(x_below_zero)))

    if x_diff_below > x_diff_above:
        # print('signal is upside down')
        x = -x
        if plot:
            fig = plt.figure(figsize=(4, 3))
            plt.plot(x[fs * 3600: fs * 3610])
            num = 0
            full_filename = 'E:\\Arc_study\\figures\\PPG\\'
            while os.path.isfile('{}_{}.png'.format(full_filename, num)): num += 1
            plt.savefig('{}_{}.png'.format(full_filename, num), transparent=True, dpi=300,
                        bbox_inches='tight')
    return x


def log(x, fs):
    return np.log(x)

def log_plus_one(x, fs):
    return np.log(x + 1)

def rolling_autocorr(x, fs, window):
    x = x[4000*fs: int(fs*4120)]
    x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))
    aut_window = (fs * window + 1) if (fs * window) % 2 == 0 else (fs * window)
    x_pd = pd.Series(x_)
    x_out = x_pd.rolling(aut_window).apply(lambda x: x.autocorr(), raw=False)

    fig = plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(x)
    plt.show()
    fig = plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(x_out)
    plt.show()
    k = 1
    return x_out


def set_min(x, fs, new_minimum=0):
    x_norm = x - np.min(x) + 0.0000000001 + new_minimum
    return x_norm



def ACC_low_amplitude_segmentation(x, fs, tolerance=0.1):

    x_low_amplitude = np.zeros((x.size))
    x_vec_mag = abs(x)
    x_low_amplitude[x_vec_mag < tolerance] = x[x_vec_mag < tolerance]

    return x_low_amplitude


def ACC_high_amplitude_segmentation(x, fs, tolerance=0.1):
    x_low_amplitude = np.zeros((x.size))
    x_vec_mag = abs(x)
    x_low_amplitude[x_vec_mag > tolerance] = x[x_vec_mag > tolerance]

    return x_low_amplitude

'''
def ENMO(x, fs):
    assert(len(x.shape) == 2 & x.shape[-1] == 3)
    enmo = (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) ** (1 / 2) - 1
    return enmo.clip(a_min=0)

def ZANGLE(x, fs):
    import math
    assert (len(x.shape) == 2 & x.shape[-1] == 3)
    return math.atan(x[:, 2] / ((x[:, 0] ** 2 + x[:, 1] ** 2) ** (1 / 2))) * 180 / math.pi

def LIDS(x, fs):
    assert (len(x.shape) == 2 & x.shape[-1] == 3)
    enmo = ENMO(x, fs=32)
    enmo_mov_sum =
    activity_count =
'''

def absum2(x, fs, fs_out):
    from scipy.ndimage import maximum_filter as median_filter

    assert(fs % fs_out == 0)
    if fs == fs_out:
        return x

    # absum_fun = lambda x: np.abs(x).sum()
    num_features = 1

    window_len = fs // fs_out
    N_out = x.shape[0] // window_len  # // is floor division.
    #x_out = np.zeros((N_out, num_features))

    x = median_filter(np.abs(x), size=(window_len,))
    x_out = x[::window_len]

    return x_out


def absum(x, fs, fs_out, plot=False):


    assert(fs % fs_out == 0)
    if fs == fs_out:
        return x

    absum_fun = lambda x: x.mean()
    num_features = x.shape[-1]

    window_len = fs // fs_out
    N_out = x.shape[0] // window_len  # // is floor division.
    x_out = np.zeros((N_out, num_features))

    for f in range(num_features):
        for n in range(N_out):
            x_win = x[n * window_len: (n + 1) * window_len, f]
            x_out[n, f] = absum_fun(x_win)

    if plot:
        fig = plt.figure(figsize=(6, 0.75))
        ax = fig.add_axes([0, 0, 1, 1])

        ax.plot(x, linewidth=0.2)
        plt.show()


        fig = plt.figure(figsize=(6, 0.75))
        ax = fig.add_axes([0, 0, 1, 1])

        #ax.plot(x_out, linewidth=.2)
        #plt.show()
        #k = 1
    return x_out

def total_variation_filter(x, fs, plot=False):

    from statsmodels.tsa.stattools import adfuller
    from scipy.signal import detrend
    from time import time

    if len(x.shape) > 1:
        x = x[:, 0]

    def TV_filter(y, lmbd=2., Nit=200):
        J = np.zeros((Nit))
        N = y.shape[0]
        z = np.zeros((N - 1))
        alpha = 4
        T = lmbd / 2

        for k in range(Nit):
            x = y - np.concatenate((-z[:1], -np.diff(z, axis=0), z[-1:]))  # y - D' z
            J[k] = (np.abs(x - y) ** 2).sum() + lmbd * (np.abs(np.diff(x))).sum()
            z = z + 1 / alpha * np.diff(x, axis=0)  # z + 1/alpha D z
            z = np.maximum(np.minimum(z, T), -T)  # clip(z,T)

        return x, J

    if x.std() > 0.009 and x.std() < 0.011 and x.mean() < 1e-4:
        return x
    else:
        x_out, J = TV_filter(x)
    k = 1
    if plot:
        plt.plot(x[3600*32*6:3600*32*6+60*32])
        plt.plot(x_out[3600*32*6:3600*32*6+60*32])
        plt.show()

        plt.plot(J[50:])
        plt.show()
        k = 1
    return x_out

def pca_decomposition(x, fs, n_components=1):
    from sklearn.decomposition import PCA

    if x.shape[-1] < 2:
        return x

    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)


def ica_decomposition(x, fs, n_components=1):
    from sklearn.decomposition import FastICA

    if x.shape[-1] < 2:
        return x

    ica = FastICA(n_components=n_components)
    x = ica.fit_transform(x)
    # ica.mixing_ # Get estimated mixing matrix
    return x

