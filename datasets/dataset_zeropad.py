import os
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import Sequence
from joblib import Memory, Parallel, delayed
import random
from datasets import Dataset


from utils import get_h5_data, get_h5_events, semantic_formating, intersection_overlap, jaccard_overlap, any_formating
from signal_processing import regularizers, normalizers


class DatasetZeroPad(Dataset):
    """
    Same as EventDataset but with the possibility to choose the probability to get at least
    one event when retrieving a window.

    """

    def __init__(self,
                 records,
                 h5_directory,
                 signals_format,
                 window,
                 number_of_channels,
                 model_format=None,
                 events_format=None,
                 events_discard_format=[],
                 events_select_format=[],
                 prediction_resolution=1,
                 overlap=0.5,
                 minimum_overlap=0.5,
                 batch_size=64,
                 transformations_factor=0.1,
                 discard_threshold=10,
                 select_threshold=10,
                 cache_data=True,
                 val_rate=0,
                 dataset_type='',
                 load_signal_in_RAM=True,
                 use_mask=True):

        super().__init__(records=records,
                         h5_directory=h5_directory,
                         signals_format=signals_format,
                         window=window,
                         events_discard_format=events_discard_format,
                         events_select_format=events_select_format,
                         number_of_channels=number_of_channels,
                         events_format=events_format,
                         overlap=overlap,
                         prediction_resolution=prediction_resolution,
                         minimum_overlap=minimum_overlap,
                         batch_size=batch_size,
                         transformations_factor=transformations_factor,
                         discard_threshold=discard_threshold,
                         select_threshold=select_threshold,
                         val_rate=val_rate,
                         dataset_type=dataset_type,
                         load_signal_in_RAM=True,
                         cache_data=cache_data,
                         model_format=model_format,
                         use_mask=use_mask)

        for record in self.records:
            for signal_name, signal_dict in self.signals_format.items():
                signal = self.signals[record]['data'][signal_name]
                signal_duration = signal.shape[0] / signal_dict['fs_post'] # in seconds

                # zero-pad signal to max window duration
                if signal_duration < window:
                    pad_size = int((window - signal_duration) * signal_dict['fs_post'])
                    self.signals[record]['data'][signal_name] = np.concatenate(
                        (signal, np.zeros((pad_size, signal.shape[-1]))), axis=0)
                    self.events_discard[record]['discard']['data'] = np.concatenate(
                        (self.events_discard[record]['discard']['data'], np.array([[signal_duration], [pad_size]])), axis=1)
            k = 1
