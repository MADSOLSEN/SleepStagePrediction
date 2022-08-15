import numpy as np
from matplotlib import pyplot as plt
import sys
import heartpy as hp
epsilon = sys.float_info.epsilon
from time import time
import hrvanalysis as aura_hrv


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


# Collector instances
# ===================

def collect_signal_features(x, fs, window, window_ext):
    feature_dict = {}
    for ext in window_ext:
        feature_dict.update({
            'energy_{}'.format(ext): {'fun': energy, 'ext': ext * fs, 'num_feat': 1},
            'SDNN_{}'.format(ext): {'fun': STD, 'ext': ext * fs, 'num_feat': 1},
            'SDSD_{}'.format(ext): {'fun': STDSTD, 'ext': ext * fs, 'num_feat': 1},
            'range_{}'.format(ext): {'fun': range_, 'ext': ext * fs, 'num_feat': 1},
            'q10_{}'.format(ext): {'fun': quantile(q=.10), 'ext': ext * fs, 'num_feat': 1},
            'q25_{}'.format(ext): {'fun': quantile(q=.25), 'ext': ext * fs, 'num_feat': 1},
            'q75_{}'.format(ext): {'fun': quantile(q=.75), 'ext': ext * fs, 'num_feat': 1},
            'q90_{}'.format(ext): {'fun': quantile(q=.90), 'ext': ext * fs, 'num_feat': 1},
            'VLF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.0033, high=0.04), 'ext': ext * fs, 'num_feat': 1},
            'LF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15), 'ext': ext * fs, 'num_feat': 1},
            'HF_{}'.format(ext): {'fun': frequency(window=window * fs, fs=fs, low=0.15, high=0.4), 'ext': ext * fs, 'num_feat': 1}
        })

    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    return fc.extract_features(x)


def FM_features(x, window, fs, window_ext=0):
    feature_dict = {
        'VLF': {'fun': frequency(window=window * fs, fs=fs, low=0.0033, high=0.04, normalize_interval=None), 'ext': 120 * fs, 'num_feat': 1},
        'LF': {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15, normalize_interval=None), 'ext': 120 * fs, 'num_feat': 1},
        'HF': {'fun': frequency(window=window * fs, fs=fs, low=0.15, high=0.5, normalize_interval=None), 'ext': 120 * fs, 'num_feat': 1},
        'LF_HF': {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.15, normalize_interval=[0.15, 0.5]), 'ext': 120 * fs, 'num_feat': 1},
        'mean_energy': {'fun': energy, 'ext': 0, 'num_feat': 1},
        'HR': {'fun': HR, 'ext': 0, 'num_feat': 1}
    }
    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    features = fc.extract_features(x)
    #feature = np.zeros((features.shape[0], 5))
    #feature[:, 0] = (features[:, 0] / (features[:, 2] + epsilon))  / ((features[:, 1] / (features[:, 2] + epsilon)) + epsilon)
    return features


def ACC_features(x, window, fs, window_ext=0):
    feature_dict = {
        'LF': {'fun': frequency(window=window * fs, fs=fs, low=0.04, high=0.5, normalize_interval=None), 'ext': 120 * fs, 'num_feat': 1},
        'HF': {'fun': frequency(window=window * fs, fs=fs, low=5, high=14, normalize_interval=None), 'ext': 120 * fs, 'num_feat': 1}, # [0.004, 14]
        'mean_energy': {'fun': energy, 'ext': 0, 'num_feat': 1},
        'activity': {'fun': activity, 'ext': 0, 'num_feat': 1},
        'mobility': {'fun': mobility, 'ext': 0, 'num_feat': 1},
        'complexity': {'fun': complexity, 'ext': 0, 'num_feat': 1},
    }
    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    feature = fc.extract_features(x)
    return feature


def collect_aura_features(x, window, window_ext, fs):
    feature_dict = {}
    for ext in window_ext:
        feature_dict.update({
            'hearpy_{}'.format(ext): {'fun': get_aura_features(fs=fs), 'ext': ext * fs, 'num_feat': 23}, # 32},
        })

    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    return fc.extract_features(x)


def collect_hrv_features(x, window, window_ext, fs):
    feature_dict = {}
    for ext in window_ext:
        feature_dict.update({
            'hearpy_{}'.format(ext): {'fun': heartpy_fun(fs=fs), 'ext': ext * fs, 'num_feat': 13},
        })

    fc = feature_collector(x, fs, window, feature_dict=feature_dict)
    return fc.extract_features(x)


# FEATURES
# ========

# HRV ANALYSIS FROM AURA GROUP
# ============================
# FROM: https://github.com/Aura-healthcare/hrvanalysis
def get_aura_features(fs):
    def aura_features(x):
        features = []
        rr = RR_(x, fs)
        rr = rr[rr > 0.15]
        # TODO - rr = RR_intervals(x, fs)

        if np.sum(rr) != 0:
            try:
                features += [f for f in aura_hrv.get_time_domain_features(nn_intervals=rr).values()]  # 16
            except:
                features += [0] * 16

            try:
                features += [f for f in aura_hrv.get_poincare_plot_features(nn_intervals=rr).values()]  # 3
            except:
                features += [0] * 3

            try:
                features += [f for f in aura_hrv.get_sampen(nn_intervals=rr).values()]  # 1
            except:
                features += [0] * 1

            try:
                features += [f for f in aura_hrv.get_csi_cvi_features(nn_intervals=rr).values()]  # 3
            except:
                features += [0] * 3

            # features += [f for f in aura_hrv.get_frequency_domain_features(nn_intervals=rr, sampling_frequency=fs).values()]
            # features += [f for f in aura_hrv.get_geometrical_features(nn_intervals=rr).values()]
            features = [0 if (f is None) or (np.isnan(f)) or (np.isinf(f)) else f for f in features]
        else:
            features = [0] * 23
        return features
    return aura_features


# Heartpy
# =======
# From: https://github.com/paulvangentcom/heartrate_analysis_python
def heartpy_fun(fs):
    def hearpy_fun(x):
        try:
            # Maybe do initial normalization...
            wd, m = hp.process(x, fs)
            m = list(m.values())
        except:
            m = [0]*13
        return m
    return hearpy_fun

def RR_intervals(x, fs):
    try:
        wd, m = hp.process(x, fs)
        return wd['RR_list']
    except:
        return None


# Sample RR from HRV signal
def RR_(x, fs):
    return x[::fs]


# time domain features
# ==================

def energy(x):
    return np.mean(np.abs(x) ** 2)

def HR(x):
    return 60 / (x.mean() + 1e-9)

def RMSSD(x):
    return np.sqrt(np.mean(np.diff(x) ** 2))

def STDSTD(x):
    return np.std(np.diff(x))

def STD(x):
    return np.std(x)

def range_(x):
    return np.diff([np.min(x), np.max(x)])

def quantile(q=.5):
    def quantile(x):
        return np.quantile(x, q=q)
    return quantile

def activity(x):
    return np.var(x)

def mobility(x):
    return np.sqrt( activity(np.diff(x)) / (activity(x) + 1e-9) )

def complexity(x):
    return np.sqrt( mobility(np.diff(x)) / (mobility(x) + 1e-9))

# frequency domain features
# ===================
def frequency(window, fs, low, high, normalize_interval=None):

    nfft = int(2 ** np.ceil(np.log2(window)))
    f = np.arange(start=0, stop=fs / 2 + fs / nfft, step=fs / nfft)

    def frequency(x):
        x_fft = np.fft.fft(x * np.hanning(x.shape[0]), n=nfft)
        x_fft = (1 / (fs * nfft)) * np.abs(x_fft[0: nfft // 2 + 1]) ** 2
        x_fft = np.log(x_fft + 1)
        out = np.sum(x_fft[(f > low) & (f <= high)])
        if normalize_interval is not None and np.sum(x_fft) > 0:
            out = out / (np.sum(x_fft[(f > normalize_interval[0]) & (f <= normalize_interval[1])]) + epsilon)
        return out
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
        E[n] = np.sum(np.abs(x_current[x_current > min_thresh]))

        if n > 3:
            t = n - 2
            feature[t] = E[t] + 0.2 * (E[t - 1] + E[t + 1]) + 0.04 * (E[t - 2] + E[t + 2])

    return feature

# Arousal index - likelihood ratios:
def LR(x):

    Limits = [
        [-3.4, -2.8, -2.6, -2.7, -2.8],
        [-1.6, -1.1, -1.0, -1.0, -1.1],
        [-0.6, 0.0, 0.1, 0.2, 0.1],
        [0.2, 1.0, 1.4, 1.5, 1.4],
        [1.1, 2.1, 2.7, 3.1, 3.0],
        [2.0, 3.5, 4.6, 5.3, 5.3],
        [3.3, 5.3, 7.1, 8.2, 8.8],
        [5.3, 7.8, 10.1, 11.8, 12.6],
        [8.8, 12.1, 14.7, 16.3, 17.4]
    ]


    LikelihoodRatios = [
        [0.5361, 0.2216, 0.1469, 0.1447, 0.2302],
        [0.5456, 0.2145, 0.1508, 0.0863, 0.1071],
        [0.4153, 0.3117, 0.1610, 0.1212, 0.1475],
        [0.5675, 0.3672, 0.2395, 0.2376, 0.2053],
        [0.5622, 0.5299, 0.5021, 0.4836, 0.3760],
        [0.8710, 1.2317, 1.1652, 1.2566, 1.0231],
        [1.0482, 1.8880, 3.1445, 3.7838, 3.6601],
        [1.7931, 3.6340, 7.1163, 11.9273, 15.3023],
        [4.0993, 7.8734, 20.4848, 42.8125, 38.5556],
        [5.7642, 16.3659, 33.0952, 57.7500, 100.0000]
    ]

    HR = 1. / (x) * 60 # Must be in bpm.FS dependent!











