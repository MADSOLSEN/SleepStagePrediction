import numpy as np
import h5py
from time import time
from signal_processing import preprocessing
from matplotlib import pyplot as plt


def get_h5_data(filename, signals):
    with h5py.File(filename, 'r') as h5:
        # print(filename, [signal for signal in signals.keys()])
        # t1 = time()
        channels = {}
        # channels = {signal_name: [] for signal_name in signals.keys()}
        for signal_name, signal in signals.items():
            x = h5[signal['h5_path']][:]
            channel = []
            for idx in signal['channel_idx']:
                x_ = x[:, idx]
                fs = h5[signal['h5_path']].attrs['fs']
                for preproc in signal['preprocessing']:
                    x_ = preprocessing[preproc['type']](x=x_,
                                                        fs=fs,
                                                        **preproc['args'])
                    if isinstance(x_, tuple):
                        x_, fs = x_

                if len(x_.shape) == 1:
                    x_ = np.expand_dims(x_, axis=-1)
                channel += [x_]

            channel = np.moveaxis(np.array(channel), 0, -1)
            # TODO - test that axis match
            if signal['add']:
                channels.update({signal_name: np.sum(channel, axis=2)})
            else:
                channels.update({signal_name : channel.reshape((channel.shape[0], np.prod(channel.shape[1:])))})
        # print('elapse time for data prep: {} sek'.format(round((time() - t1) * 100) / 100))
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

def get_h5_auxiliary(filename, labels):
    with h5py.File(filename, 'r') as h5:
        x = []
        for label in labels:
            x += [h5[label][:]]
            fs = h5[label].attrs['fs']
        x = np.moveaxis(np.array(x), 0, -1)
    if (x.ndim != 2) or (x.shape[-1] != 4):
        print(x.shape)
    return x, fs



def save_filename_to_mat(filename='mesa-sleep-0012.h5'):
    from scipy.io import savemat
    import os


    #filename = 'mesa-sleep-0001.h5'
    save_dir = 'E:\Arc_study\\figures\\analyzeSSoutput'
    old_dir = 'E:\datasets\mesa\processed_data\h5\model_output'
    new_dir = 'E:\datasets\mesa\processed_data\h5\SleepStagePrediction_ver2\databases\ACC_fft_PPG_fft_ResUSleep\with_mesa_WL_30720_ext_5_bs_2\model_output'
    events = ['wake', 'light', 'deep', 'rem']

    old = get_h5_auxiliary(filename=os.path.join(old_dir, filename), labels=events)
    new = get_h5_auxiliary(filename=os.path.join(new_dir, filename), labels=events)

    old_dict = {'array': old[0], 'fs': old[1]}
    new_dict = {'array': new[0], 'fs': new[1]}

    savemat(file_name=os.path.join(save_dir, 'old_{}.mat'.format(filename[:-3])), mdict=old_dict)
    savemat(file_name=os.path.join(save_dir, 'new_{}.mat'.format(filename[:-3])), mdict=new_dict)

#save_filename_to_mat()