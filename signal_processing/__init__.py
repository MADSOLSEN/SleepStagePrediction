from signal_processing.normalization import adaptive_soft, zscore_log, min_max, soft, median, clip, no_normalization, \
    complete_zscore, median_filter, divide, butter_bandpass_filter, zscore_norm, log, rolling_autocorr, set_min, \
    downsample, diff, butter_highpass_filter, clip_by_iqr, iqr_normalization, ACC_low_amplitude_segmentation, \
    ACC_high_amplitude_segmentation, log_plus_one
from signal_processing.regularization import MagWarp, Jitter, Scaling, Inverting, TimeWarp, image_translation, plot_stuff, time_mask, freq_mask, shuffle_features
from signal_processing.postprocessing import post_processing
from signal_processing.spectrogram import cal_mfcc, cal_psd, cal_logfbank, plot_cwt, cal_autocorrelation_psd, cal_wavelet, cal_wavelet_new
from signal_processing.feature_extraction import acc_feature, collect_signal_features, FM_features, ACC_energy, collect_aura_features, collect_hrv_features

preprocessing = {
    "complete_zscore": complete_zscore,
    "logfbank": cal_logfbank,
    "mfcc": cal_mfcc,
    'adaptive_soft': adaptive_soft,
    'zscore_log': zscore_log,
    'min_max': min_max,
    'soft': soft,
    'median': median,
    'clip': clip,
    'no_normalization': no_normalization,
    'median_filter': median_filter,
    'divide': divide,
    'butter_bandpass_filter': butter_bandpass_filter,
    'zscore': zscore_norm,
    'log': log,
    "cal_psd": cal_psd,
    'cal_autocorrelation_psd': cal_autocorrelation_psd,
    'cal_wavelet': cal_wavelet,
    'cal_wavelet_new': cal_wavelet_new,
    'set_min': set_min,
    'collect_signal_features': collect_signal_features,
    'acc_manual_feature': acc_feature,
    'FM_features': FM_features,
    'ACC_energy': ACC_energy,
    'collect_aura_features': collect_aura_features,
    'collect_hrv_features': collect_hrv_features,
    'downsample': downsample,
    'diff': diff,
    'butter_highpass_filter': butter_highpass_filter,
    'clip_by_iqr': clip_by_iqr,
    'iqr_normalization': iqr_normalization,
    'ACC_low_amplitude_segmentation': ACC_low_amplitude_segmentation,
    'ACC_high_amplitude_segmentation': ACC_high_amplitude_segmentation,
    'log_plus_one': log_plus_one
}

normalizers = {
    'complete_zscore': complete_zscore,
    'adaptive_soft': adaptive_soft,
    'zscore_log': zscore_log,
    'min_max': min_max,
    'soft': soft,
    'median': median,
    'clip': clip,
    'no_normalization': no_normalization,
    'divide': divide,
    'butter_bandpass_filter': butter_bandpass_filter,
    'butter_highpass_filter': butter_highpass_filter,
    'zscore': zscore_norm,
    'log': log,
    #'set_min': set_min,
}

regularizers = {
    'Inverting': Inverting,
    'Jitter': Jitter,
    'Scaling': Scaling,
    'MagWarp': MagWarp,
    'TimeWarp': TimeWarp,
    'image_translation': image_translation,
    'plot_stuff': plot_stuff,
    'time_mask': time_mask,
    'freq_mask': freq_mask,
    'shuffle_features': shuffle_features
}

__all__ = ['post_processing', 
           'adaptive_soft',
           'zscore_log',
           'min_max',
           'soft',
           'median',
           'clip',
           'no_normalization',
           'plot_cwt',
           'rolling_autocorr',
           'butter_bandpass_filter']