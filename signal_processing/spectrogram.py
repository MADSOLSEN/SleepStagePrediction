import sys
import os
from scipy.signal import stft, cwt, spectrogram
from scipy import signal
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from python_speech_features import mfcc
from python_speech_features import logfbank, fbank
import scipy.io.wavfile as wav
from time import time
from utils.buffer import buffer_indexes
from signal_processing import zscore_norm
from matplotlib import gridspec
from matplotlib.colors import LogNorm

def apply_max_len(x, fs):
    maxLen = 3600 * 9
    if x.size / fs > maxLen:
        return x[:maxLen*fs]
    return x


def cal_autocorrelation_psd(x, fs, window, noverlap, nfft, f_min, f_max, normalize=True):

    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=np.std(x) * sys.float_info.epsilon, size=(x_.shape))

    nfft = int(2 ** np.ceil(np.log2(window)))
    buf_idx = buffer_indexes(x_, window, noverlap / window)
    S = np.zeros((buf_idx.shape[0], nfft // 2 + 1))
    f = np.arange(start=0, stop=fs/2 + fs/nfft, step=fs/nfft)
    x_zp = np.zeros((window * 2,))

    for n in range(buf_idx.shape[0]):
        if buf_idx[n, 0] >= window // 2 and buf_idx[n, -1] <= x_.shape[0] - window // 2:
            if normalize:
                x_ext = zscore_norm(x_[buf_idx[n, 0] - window // 2: buf_idx[n, -1] + window // 2 + 1], fs=fs)
                x_zp[window // 2: -window // 2] = zscore_norm(x_[buf_idx[n, :]], fs=fs)
            else:
                x_ext = x_[buf_idx[n, 0] - window // 2: buf_idx[n, -1] + window // 2 + 1]
                x_zp[window // 2: -window // 2] =  x_[buf_idx[n, :]]
            x_corr = np.correlate(x_ext, x_zp, mode='full')
            x_corr = x_corr[window // 2 : -window // 2]
            x_fft = np.fft.fft(x_corr * np.hanning(x_corr.shape[0]), n=nfft)
            x_fft = (1 / (fs * nfft)) * np.abs(x_fft[0: nfft // 2 + 1]) ** 2
            S[n, :] = np.log(x_fft + sys.float_info.epsilon)
    S = S[:, (f > f_min) & (f <= f_max)]
    #assert (round(S.shape[0] / (fs / (window - noverlap))) >= round(x.size / fs) - 1 and round(
    #    S.shape[0] / (fs / (window - noverlap))) <= round(x.size / fs) + 1)

    assert (np.sum(np.isinf(S)) == 0)
    assert (np.sum(np.isnan(S)) == 0)

    return S

def cal_psd_old(x, fs, window, noverlap, nfft, f_min, f_max, plot_flag=True):

    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x_.shape))

    t1 = time()
    S, f, t, ax = plt.specgram(x=x_,
                               Fs=fs,
                               window=np.blackman(window),
                               NFFT=window,
                               noverlap=noverlap)

    print(time() - t1)
    S = S[(f > f_min) & (f <= f_max), :]
    S = np.swapaxes(S, axis1=1, axis2=0)
    S = np.log(S + sys.float_info.epsilon)
    #assert (round(S.shape[0] / (fs / (window - noverlap))) >= round(x.size / fs) - 1 and round(
    #    S.shape[0] / (fs / (window - noverlap))) <= round(x.size / fs) + 1)

    if plot_flag:
        plot_histogram(S)
        l = 1
        start = int(3600 * 3 + 60)
        stop = int(3600 * 3 + 300)
        plot_spectrogram(S[start:stop], f_min=f_min, f_max=f_max, t_min=0.0, t_max=x.shape[0]//fs, text='spectrogram')
    k = 1
    return S

def cal_psd(x, fs, window, noverlap, nfft, f_min, f_max, f_sub=1, plot_flag=False):
    from scipy.ndimage import maximum_filter as maxfilt
    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x_.shape))

    f, t, S = spectrogram(x=x_,
                          fs=fs,
                          window=np.blackman(window),
                          nperseg=window,
                          noverlap=noverlap,
                          nfft=nfft)

    S = S[(f > f_min) & (f <= f_max), :]
    S = maxfilt(np.abs(S), size=(f_sub, 1))
    S = S[::f_sub, :]
    S = np.swapaxes(S, axis1=1, axis2=0)
    S = np.log(S + sys.float_info.epsilon)
    if S.shape[1] != 24:
        k = 1
    #assert (round(S.shape[0] / (fs / (window - noverlap))) >= round(x.size / fs) - 1 and round(
    #    S.shape[0] / (fs / (window - noverlap))) <= round(x.size / fs) + 1)

    assert (np.sum(np.isinf(S)) == 0)
    assert (np.sum(np.isnan(S)) == 0)

    if plot_flag:
        plot_histogram(S)
        l = 1
        start = int(3600 * 3 + 60)
        stop = int(3600 * 3 + 300)
        plot_spectrogram(S[start:stop], f_min=f_min, f_max=f_max, t_min=0.0, t_max=x.shape[0] // fs, text='spectrogram')
    return S


def cal_logfbank(x, fs, window, noverlap, nfft, numcep, f_min, f_max, plot_flag=False):

    x = apply_max_len(x, fs)

    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

    S = fbank(signal=x_,
              samplerate=fs,
              winlen=(window / fs),
              winstep=(window - noverlap) / fs,
              nfilt=numcep,
              nfft=nfft,
              winfunc=np.hanning)
    S = np.log(S[0] + sys.float_info.epsilon)
    assert (round(S.shape[0] / (fs / (window - noverlap))) == round(x.size / fs))

    if plot_flag:
        plot_histogram(S)
        plot_spectrogram(S, f_min=f_min, f_max=f_max, t_min=0.0, t_max=x.shape[0] // fs, text='mfcc')
    return S


def cal_mfcc(x, fs, window, noverlap, nfft, numcep, f_min, f_max, plot_flag=False):

    x = apply_max_len(x, fs)

    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x_.shape))

    S = mfcc(signal=x_,
                samplerate=fs,
                winlen=(window / fs),
                winstep=(window - noverlap) / fs,
                numcep=numcep,
                nfilt=numcep * 2,
                nfft=nfft,
                preemph=0.97,
                ceplifter=22,
                winfunc=np.hanning)
    assert (round(S.shape[0] / (fs / (window - noverlap))) == round(x.size / fs))

    if plot_flag:
        plot_spectrogram(S, f_min=f_min, f_max=f_max, t_min=0.0, t_max=x.shape[0]//fs, text='mfcc')
    return S

def cal_wavelet(x, fs, window, noverlap, nfft, f_min, f_max, plot_flag=False, sub=16):

    # add noise
    x = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

    # wavelet
    wavelet = 'morl'
    # scales = np.arange(2, 2 + 64, step=1)
    scales = np.logspace(start=1, stop=7, num=64, base=2)
    k = pywt.scale2frequency(wavelet, scales) * fs
    coefs, frequencies = pywt.cwt(data=x, scales=scales, wavelet=wavelet, sampling_period=1 / fs)
    coefs = np.flipud(coefs)
    coefs = np.swapaxes(coefs, 1, 0)
    #coefs = np.log2(coefs - np.min(coefs) + sys.float_info.epsilon)

    fs_new = fs / sub

    assert (np.sum(np.isinf(coefs)) == 0)
    assert (np.sum(np.isnan(coefs)) == 0)

    if plot_flag:
        # cal_test_spec(x, fs, window, noverlap, nfft, f_min, f_max)

        start = int(3600 * 6.5)
        stop = int(3600 * 6.5 + 3600)
        t = np.arange(start=start, stop=stop, step=1 / fs_new)

        coef_in = coefs[int(start * fs_new):int(stop * fs_new), :]

        plot_cwt(coef_in, t=t, f=frequencies,
                 text=wavelet, x=x[int(start * 32):int(stop * 32)])
        k = 1

    return coefs


def cal_wavelet_new(x, fs, window, noverlap, nfft, f_min, f_max, plot_flag=False, sub=16):
    from scipy.ndimage import maximum_filter as maxfilt
    from scipy.ndimage import median_filter as medfilt

    # add noise
    x = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

    # wavelet
    wavelet = 'morl'
    # scales = np.arange(2, 2 + 64, step=1)
    scales = np.logspace(start=1, stop=7, num=64, base=2)
    k = pywt.scale2frequency(wavelet, scales) * fs
    coefs, frequencies = pywt.cwt(data=x, scales=scales, wavelet=wavelet, sampling_period=1 / fs)
    coefs = np.flipud(coefs)
    coefs = np.swapaxes(coefs, 1, 0)
    coefs = maxfilt(np.abs(coefs), size=(sub, 1))
    coefs = coefs[::sub, :]
    #coefs = np.log2(coefs - np.min(coefs) + sys.float_info.epsilon)

    fs_new = fs / sub

    assert (np.sum(np.isinf(coefs)) == 0)
    assert (np.sum(np.isnan(coefs)) == 0)

    if plot_flag:
        # cal_test_spec(x, fs, window, noverlap, nfft, f_min, f_max)

        start = int(3600 * 6.5)
        stop = int(3600 * 6.5 + 3600)
        t = np.arange(start=start, stop=stop, step=1 / fs_new)

        coef_in = coefs[int(start * fs_new):int(stop * fs_new), :]

        plot_cwt(coef_in, t=t, f=frequencies,
                 text=wavelet, x=x[int(start * 32):int(stop * 32)])
        k = 1

    return coefs

def cal_test_spec(x, fs, window,  noverlap, nfft, f_min, f_max):
    start = int(3600*6.5)
    stop = int(3600*6.5 + 60)
    spec = cal_psd(x, fs, window, noverlap, nfft, f_min, 3, plot_flag=False)
    spec_auto = cal_autocorrelation_psd(x, fs, window, noverlap, nfft, f_min, 3)
    coefs = cal_wavelet(x, fs, window, noverlap, nfft, f_min, f_max, plot_flag=False)
    S = [spec[start:stop, :], spec_auto[start:stop, :], coefs[int(start * 32):int(stop * 32), :]]
    plot_spectrogram_all(S, f_min=f_min, f_max=4, t_min=0, t_max=spec.shape[0]//fs, text='wavelet_8_mexh', x=x[int(start * 32):int(stop * 32)])

def plot_cwt(S, t, f, x=None, text=''):

    save_path = 'E:\\Arc_study\\figures\\cwt'
    fig = plt.figure(figsize=(8,1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fontsize = 6

    # signals[key] = np.exp(signals[key])
    S = np.swapaxes(S, axis1=1, axis2=0)
    S = np.flipud(S)
    # S = np.exp(S)
    #ax.matshow(S, cmap=cm.hot, norm=LogNorm(vmin=np.amin(f), vmax=np.amax(f)))

    ax.imshow(S, extent=(np.amin(t), np.amax(t), np.amin(f), np.amax(f)), aspect='auto', vmin=np.quantile(S, q=0.05), vmax=np.quantile(S, q=0.99))

    # x axis
    #xaxis = np.arange(start=np.amin(t), stop=np.amax(t), step= (np.amax(t) - np.amin(t)) / 16)
    xaxis = t[::t.shape[0] // 8]
    ax.xaxis.set_ticks(xaxis)
    ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
    plt.xlabel('time (min)', fontsize=fontsize)
    plt.xlim([np.amin(t), np.amax(t)])  # / 60])

    # y axis
    yticks = np.arange(start=f[0], stop=f[-1], step=(f[-1] - f[0]) / 8)
    yaxis = f[::f.shape[0]//8]
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
    ax.yaxis.set_ticklabels([int(y*100*60) / 100 for y in yaxis], fontsize=fontsize)
    plt.ylabel(r'freq (Hz)', size=fontsize, rotation=90)

    num = 0
    while os.path.isfile('{}\\{}_{}.png'.format(save_path, text, num)): num += 1
    plt.savefig('{}\\{}_{}.png'.format(save_path, text, num), transparent=True, dpi=300, bbox_inches='tight')
    k = 1

    if x is not None:

        fig = plt.figure(figsize=(8, 1.5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fontsize = 6

        t = np.arange(start=np.amin(t), stop=np.amax(t) + 1/32, step=1/32)

        sig_norm = (x - x.min()) / (x.max() - x.min() + sys.float_info.epsilon) * 2 - 1
        ax.plot(t, sig_norm, linewidth=.4)

        xaxis = t[::t.shape[0] // 16]
        ax.xaxis.set_ticks(xaxis)
        ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
        plt.xlabel('time (min)', fontsize=fontsize)
        plt.xlim([t[0], t[-1]])  # / 60])

        # y axis
        yaxis = [-1, 0, 1]
        # ax.yaxis.set_ticks(yaxis)
        ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
        ax.yaxis.set_ticklabels(yaxis, fontsize=fontsize)
        plt.ylabel(r'normalized amplitude', size=fontsize, rotation=90)
        plt.ylim([-2, 2])

        ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)

        num = 0
        while os.path.isfile('{}\\{}_{}.png'.format(save_path, 'sig', num)): num += 1
        plt.savefig('{}\\{}_{}.png'.format(save_path, 'sig', num), transparent=True, dpi=300, bbox_inches='tight')
    k = 1


def plot_spectrogram(S, f_min, f_max, t_min, t_max, x=None, text=''):


    fig = plt.figure(figsize=(25, 10), dpi=200)
    fig.suptitle('analysis curves', fontsize=14)

    s = S  # np.swapaxes(S, axis1=1, axis2=0)
    s = np.flipud(s)
    if np.min(s) < 0:
        s = np.exp(s)
    plt.imshow(s, extent=(t_min, t_max, f_min, f_max), cmap=cm.hot, aspect='auto')
    plt.title(text)
    plt.ylabel(r'freq (Hz)', size=12)
    plt.xlabel(r'time (s)', size=12)
    plt.savefig('{}\\spec_{}.pdf'.format('E:\\Arc_study\\figures', text))
    l = 1


def plot_spectrogram_all(S, f_min, f_max, t_min, t_max, x=None, text=''):

    grid_len = len(S) if x is None else len(S) + 1

    gs = gridspec.GridSpec(grid_len, 1)
    fig = plt.figure(figsize=(120, 10 * grid_len), dpi=200)
    fig.suptitle('analysis curves', fontsize=14)

    for idx, s in enumerate(S):

        ax = fig.add_subplot(gs[idx])
        s = np.swapaxes(s, axis1=1, axis2=0)
        s = np.flipud(s)
        if np.min(s) < 0:
            s = np.exp(s)
        ax.imshow(s, extent=(t_min, t_max, f_min, f_max), cmap=cm.hot, aspect='auto')
        ax.set_title(text)
        ax.set_ylabel(r'freq (Hz)', size=12)
        ax.set_xlabel(r'time (s)', size=12)

    if x is not None:
        ax = fig.add_subplot(gs[grid_len - 1])
        ax.plot(x)

    plt.savefig('{}\\spec_{}.pdf'.format('E:\\Arc_study\\figures', text))
    k = 1

def plot_histogram(S, bins=100, text=''):
    plt.figure(figsize=(15, 7), dpi=150)
    S = np.reshape(S, (S.size))
    n, bins, patches = plt.hist(S, 50)
    plt.savefig('{}\\hist_{}.pdf'.format('E:\\Arc_study\\figures', text))
    # plt.show()