import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from signal_processing import DA_MagWarp, DA_Jitter, DA_Scaling, DA_Inverting, DA_TimeWarp
from datasets import get_train_validation_test, Dataset, BalancedDataset
from models import aggregated_residual_model, transfer_model, reconfigure_model, get_model_activation, predict_dataset, train

seed = 2020
h5_directory = 'D:\\datasets\\mesa\\processed_data\\h5\\'
# h5_directory = 'D:\\datasets\\mesa\\processed_data\\h5_balanced\\'
# h5_directory = 'D:\\datasets\\Amazfit\\processed_data\\h5\\'
model_directory = 'D:\\Arc_study\\models\\'
tensorboard_directory = 'D:\\Arc_study\\tensorboard\\'
training_log_directory = 'D:\\Arc_study\\log\\'


train_recs, validation_recs, test_recs = get_train_validation_test(h5_directory,
                                                                   percent_test=3, # 20
                                                                   percent_validation=3, # 20
                                                                   seed=seed)

# balance filesnames:
# train__recs_balanced = train + [tr for tr in train if tr[:4] == 'STNF']*2
train_recs = train_recs[:50]

print("Number of records train:", len(train_recs))
print("Number of records validation:", len(validation_recs))
print("Number of records test:", len(test_recs))

window = 240  # window duration in seconds

signals_format = [
    {
        'h5_path': 'ppg',
        'fs': 32
    },
    {
        'h5_path': 'emg_combined',
        'fs': 32
    }
]

events_format = [
    {
        'name': 'SDB',
        'h5_path': 'SDB',
        'probability': 0.75
    }
]


dataset_parameters = {
    "h5_directory": h5_directory,
    "signals_format": signals_format,
    "window": window,
    "fs": 32,
    "events_format": events_format,
    "overlap": 0.5,
    "minimum_overlap": 0.75,
    "batch_size": 64,
    "transformations": None,
    "mask_threshold": 0.2
}


train_dataset = BalancedDataset(records=train_recs, **dataset_parameters)
validation_dataset = Dataset(records=validation_recs, **dataset_parameters)


for idx in range(train_dataset.__len__()):
    signal, events = validation_dataset.__getitem__(idx)

    ts_signal = np.linspace(1 / train_dataset.fs, train_dataset.window, signal.shape[1])
    ts_event = np.linspace(1, train_dataset.window, train_dataset.window)

    plt.figure(figsize=(10 * 2, 2 * train_dataset.number_of_channels), dpi=600)
    gs = gridspec.GridSpec(train_dataset.number_of_channels, 1)
    gs.update(wspace=0., hspace=0.)
    for batch_num in range(signal.shape[0]):
        if any(events[batch_num, :, 0]):
            for channel_num in range(len(train_dataset.signals_format)):
                ax = plt.subplot(gs[channel_num, 0])
                # ax.set_ylim(-0.55, 0.55)
                ax.plot(ts_signal, signal[batch_num, :, channel_num], alpha=.3, linewidth=.3)

                for event_num in range(len(train_dataset.events_format)):
                    ax.plot(ts_event, events[batch_num, :, event_num]*5,
                            alpha=.5,
                            linewidth=.3,
                            color="C{}".format(event_num))
            plt.savefig('D:\Arc_study\\figures\\Arc2_test' + str(batch_num) + '.pdf')
            plt.close()




for n in range(train_dataset.__len__()):
    signal, events = train_dataset.__getitem__(n)
    ts_signal = np.linspace(1 / train_dataset.fs, train_dataset.window, signal.shape[1])
    x1 = DA_Jitter(signal[0, :, :])
    x2 = DA_Scaling(signal[0, :, :])
    x3 = DA_Inverting(signal[0, :, :])
    x4 = DA_MagWarp(signal[0, :, :])
    x5 = DA_TimeWarp(signal[0, :, :])

    plt.figure(figsize=(30, 16), dpi=600)

    plt.subplot(3, 2, 1)
    plt.plot(ts_signal, signal[0, :, :], linewidth=.3)
    plt.title('raw')

    plt.subplot(3, 2, 2)
    plt.plot(ts_signal, x1, linewidth=.3)
    plt.title('Jitter')

    plt.subplot(3, 2, 3)
    plt.plot(ts_signal, x2, linewidth=.3)
    plt.title('scaling')

    plt.subplot(3, 2, 4)
    plt.plot(ts_signal, x3, linewidth=.3)
    plt.title('inverting')

    plt.subplot(3, 2, 5)
    plt.plot(ts_signal, x4, linewidth=.3)
    plt.title('magwarp')

    plt.subplot(3, 2, 6)
    plt.plot(ts_signal, x5, linewidth=.3)
    plt.title('timewarp')

    plt.savefig('D:\Arc_study\\figures\\regu.pdf')
    k = 1