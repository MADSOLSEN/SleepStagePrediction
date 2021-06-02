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
    if fs is not fs_out:
        x_out = signal.resample_poly(x, up=1, down=fs // fs_out)

        if plot:
            f, t, Sxx = signal.spectrogram(x, fs)
            plt.pcolormesh(t, f, np.log2(Sxx), shading='gouraud')
            plt.show()

            f, t, Sxx = signal.spectrogram(x_out, fs_out)
            plt.pcolormesh(t, f, np.log2(Sxx), shading='gouraud')
            plt.show()

            k = 1
        return x_out
        #step = int(fs / fs_out)
        #return x[::step]
    else:
        return x


def zscore_norm(x, fs=1, axis=0):

    x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))
    x_norm = zscore(x_, axis=axis)

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
    x_med = np.ones((x.shape)) * np.median(x)

    x_pd = pd.Series(x)
    med_ = x_pd.rolling(window).median()
    x_med[int(window / 2):-int(window / 2)] = med_[window - 1:]
    x_med[:int(window / 2)] = med_[window - 1]
    x_med[-int(window / 2):] = med_[-1:]

    x_med[np.isnan(x_med)] = 0  # remove nan

    x_norm = x - x_med
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
    x_out[1:] = np.diff(x)
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

def clip_by_iqr(x, fs, threshold=20, iqr_upper=0.75):

    deviation = np.abs(x - np.median(x))
    iqr_up = np.quantile(x, iqr_upper)
    x[deviation > threshold * iqr_up] = threshold * iqr_up

    return x

def iqr_normalization(x, fs, median_window, iqr_window):
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
        return (x_ - x_med) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon)

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x, fs, median_window, iqr_window)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n], fs, median_window, iqr_window)
    return x_norm

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
