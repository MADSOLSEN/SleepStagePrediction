import numpy as np
from matplotlib import pyplot as plt
import sys
import heartpy as hp
epsilon = sys.float_info.epsilon
from time import time

class feature_collector():

    def __init__(self, x, fs, window, feature_dict={}):
        self.fs = fs
        self.window_size = window * fs
        self.n_windows = x.shape[0] // self.window_size
        self.features = np.empty((self.n_windows, 0))
        self.feature_dictionary = feature_dict

    def initiate_feature(self, num_features=1):
        features = np.zeros((self.n_windows, num_features))
        return features

    def add_to_features(self, features_to_add):
        self.features = np.append(self.features, features_to_add, axis=1)

    def window_generator(self, x, extend=0):
        for n in range(self.n_windows):
            start_idx = max(n * self.window_size - extend, 0)
            stop_idx = min((1 + n) * self.window_size + extend, x.shape[0])
            yield x[start_idx: stop_idx]

    def extract_features(self, x):
        for key, item in self.feature_dictionary.items():
            feature = self.initiate_feature(num_features=item['num_feat'])
            for n, x_win in enumerate(self.window_generator(x, extend=item['ext'])):
                feature[n, :] = item['fun'](x_win)
            self.add_to_features(features_to_add=feature)
        return self.features

# hearpy:
def hearpy_fun(fs):
    def hearpy_fun(x):
        try:
            wd, m = hp.process(x, fs)
            m = list(m.values())
        except:
            m = [0]*13
        return m
    return hearpy_fun

# time domain features
# ==================
def energy(x):
    return np.sum(np.abs(x) ** 2)

def RMSSD(x):
    return np.sqrt(np.mean(np.diff(x) ** 2))

def SDSD(x):
    return np.std(np.diff(x))

def SDNN(x):
    return np.std(x)

def range_(x):
    return np.diff([np.min(x), np.max(x)])

def quantile(q=.5):
    def quantile(x):
        return np.quantile(x, q=q)
    return quantile

# frequency features
# ===================
def frequency(window, fs, low, high):

    nfft = int(2 ** np.ceil(np.log2(window)))
    f = np.arange(start=0, stop=fs / 2 + fs / nfft, step=fs / nfft)

    def frequency(x):
        x_fft = np.fft.fft(x * np.hanning(x.shape[0]), n=nfft)
        x_fft = (1 / (fs * nfft)) * np.abs(x_fft[0: nfft // 2 + 1]) ** 2
        #x_fft = np.log2(x_fft + 0.0001)
        return np.sum(x_fft[(f > low) & (f <= high)])
    return frequency

# Cole Kripke
# ===================
def acc_feature(x, fs, window, min_thresh=0.1):

    window_size = window * fs
    n_windows = x.shape[0] // window_size
    num_features = 1

    E = np.zeros((n_windows, num_features))
    feature = np.zeros((n_windows, num_features))

    for n in range(n_windows):
        x_current = x[n * window_size: (1 + n) * window_size]
        E[n] = np.sum(x_current[x_current > min_thresh])

        if n > 3:
            t = n - 2
            feature[t] = E[t] + 0.2 * (E[t - 1] + E[t + 1]) + 0.04 * (E[t - 2] + E[t + 2])

    return feature

def hrv_extraction(x, fs, window):

    feature_dict = {}
    extensions = [0, 1 * 60, 2 * 60, 4 * 60]
    for ext in extensions:
        feature_dict.update({
            'hearpy_{}'.format(ext): {'fun': hearpy_fun(fs=fs), 'ext': ext * fs, 'num_feat': 13},
            'energy_{}'.format(ext): {'fun': energy, 'ext': ext * fs, 'num_feat': 1},
            'SDNN_{}'.format(ext): {'fun': SDNN, 'ext': ext * fs, 'num_feat': 1},
            'SDSD_{}'.format(ext): {'fun': SDSD, 'ext': ext * fs, 'num_feat': 1},
            'range_{}'.format(ext): {'fun': range_, 'ext': ext * fs, 'num_feat': 1},
            'q10_{}'.format(ext): {'fun': quantile(q=.10), 'ext': ext * fs, 'num_feat': 1},
            'q25_{}'.format(ext): {'fun': quantile(q=.25), 'ext': ext * fs, 'num_feat': 1},
            'q75_{}'.format(ext): {'fun': quantile(q=.75), 'ext': ext * fs, 'num_feat': 1},
            'q90_{}'.format(ext): {'fun': quantile(q=.90), 'ext': ext * fs, 'num_feat': 1},
            'VLF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.0033, high=0.04), 'ext': ext * fs, 'num_feat': 1},
            'LF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15), 'ext': ext * fs, 'num_feat': 1},
            'HF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.15, high=0.4), 'ext': ext * fs, 'num_feat': 1},
        })

    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    return fc.extract_features(x)


def get_all_features(x, fs, window):
    feature_dict = {}
    extensions = [0, 1 * 60, 2 * 60, 4 * 60]
    for ext in extensions:
        feature_dict.update({
            'energy_{}'.format(ext): {'fun': energy, 'ext': ext * fs, 'num_feat': 1},
            'SDNN_{}'.format(ext): {'fun': SDNN, 'ext': ext * fs, 'num_feat': 1},
            'SDSD_{}'.format(ext): {'fun': SDSD, 'ext': ext * fs, 'num_feat': 1},
            'range_{}'.format(ext): {'fun': range_, 'ext': ext * fs, 'num_feat': 1},
            'q10_{}'.format(ext): {'fun': quantile(q=.10), 'ext': ext * fs, 'num_feat': 1},
            'q25_{}'.format(ext): {'fun': quantile(q=.25), 'ext': ext * fs, 'num_feat': 1},
            'q75_{}'.format(ext): {'fun': quantile(q=.75), 'ext': ext * fs, 'num_feat': 1},
            'q90_{}'.format(ext): {'fun': quantile(q=.90), 'ext': ext * fs, 'num_feat': 1},
            'VLF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.0033, high=0.04), 'ext': ext * fs, 'num_feat': 1},
            'LF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15), 'ext': ext * fs, 'num_feat': 1},
            'HF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.15, high=0.4), 'ext': ext * fs, 'num_feat': 1},
        })

    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    return fc.extract_features(x)

def PRV_features(x, window, fs):
    feature_dict = {
        'LF': {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15), 'ext': 120 * fs},
        'HF': {'fun': frequency(window=window * fs, fs=fs, low=0.15, high=0.4), 'ext': 120 * fs},
        'T': {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.5), 'ext': 120 * fs},
    }
    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    features = fc.extract_features(x)
    feature = np.zeros((features.shape[0], 1))
    feature[:, 0] = (features[:, 0] / (features[:, 2] + epsilon))  / ((features[:, 1] / (features[:, 2] + epsilon)) + epsilon)
    return feature

def ACC_energy(x, window, fs):
    feature_dict = {
        'energy': {'fun': energy, 'ext': 0},
    }
    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    feature = fc.extract_features(x)
    return feature

# Add:
# TODO - Include raw 3D ACC as feature input...
# TODO - pNN50
# TODO - delta RR