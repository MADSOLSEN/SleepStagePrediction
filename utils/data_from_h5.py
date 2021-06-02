import numpy as np
import h5py
from time import time
from signal_processing import preprocessing
from matplotlib import pyplot as plt


def get_h5_data(filename, signals):
    with h5py.File(filename, 'r') as h5:
        print(filename, [signal for signal in signals.keys()])
        t1 = time()
        channels = {}
        # channels = {signal_name: [] for signal_name in signals.keys()}
        for signal_name, signal in signals.items():
            x = h5[signal['h5_path']][:]
            fs = h5[signal['h5_path']].attrs['fs']
            channel = []
            for idx in signal['channel_idx']:
                x_ = x[:, idx]
                for preproc in signal['preprocessing']:
                    x_ = preprocessing[preproc['type']](x=x_,
                                                        fs=fs,
                                                        **preproc['args'])
                if len(x_.shape) == 1:
                    x_ = np.expand_dims(x_, axis=-1)
                channel += [x_]

            channel = np.moveaxis(np.array(channel), 0, -1)
            if signal['add']:
                channels.update({signal_name: np.sum(channel, axis=2)})
            else:
                channels.update({signal_name : channel.reshape((channel.shape[0], np.prod(channel.shape[1:])))})
        print('elapse time for data prep: {} sek'.format(round((time() - t1) * 100) / 100))
    return channels


def get_h5_events(filename, event):
    with h5py.File(filename, 'r') as h5:
        starts = h5[event['h5_path']]['start'][:]
        durations = h5[event['h5_path']]['duration'][:]
        assert len(starts) == len(durations), 'Inconsistents event durations and starts'
        data = np.zeros((2, len(starts)))
        data[0, :] = (starts).astype(int)
        data[1, :] = (durations).astype(int)
    return data