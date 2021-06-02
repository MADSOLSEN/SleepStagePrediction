import os
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import Sequence
from joblib import Memory, Parallel, delayed
import random
np.seterr(all='raise')

from utils import get_h5_data, get_h5_events, semantic_formating, intersection_overlap, jaccard_overlap, any_formating
from signal_processing import regularizers, normalizers


class Dataset(Sequence):

    """Extract data and events from h5 files and provide efficient way to retrieve windows with
    their corresponding events.

    args
    ====

    h5_directory:
        Location of the generic h5 files.
    signals:
        The signals from the h5 we want to include together with their normalization
    events:
        The events from the h5 we want to train on
    window:
        Window size in seconds
    downsampling_rate:
        Downsampling rate to apply to signals
    records:
        Use to select subset of records from h5_directory, default is None and uses all available recordings
    n_jobs:
        Number of process used to extract and normalize signals from h5 files.
    cache_data:
        Cache results of extraction and normalization of signals from h5_file in h5_directory + "/.cache"
        We strongly recommend to keep the default value True to avoid memory overhead.
    minimum_overlap:
        For an event on the edge to be considered included in a window
    ratio_positive:
        Sample within a training batch will have a probability of "ratio_positive" to contain at least one spindle

    """

    def __init__(self,
                 records,
                 h5_directory,
                 signals_format,
                 window,
                 fs,
                 number_of_channels,
                 input_shape=None,
                 events_format=None,
                 events_discard_format=[],
                 events_select_format=[],
                 prediction_resolution=1,
                 overlap=0.5,
                 minimum_overlap=0.5,
                 batch_size=64,
                 transformations=None,
                 transformations_factor=0.1,
                 discard_threshold=10,
                 select_threshold=10,
                 cache_data=True,
                 n_jobs=1,
                 val_rate=0,
                 dataset_type='',
                 seed=2020,
                 batch_normalization={}):

        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = (window * fs, number_of_channels)

        # get number of events
        if events_format: 
            self.number_of_events = len(events_format)
            self.number_of_classes = len(events_format)
            self.event_probabilities = [event['probability'] for event in events_format]
            assert sum(self.event_probabilities) <= 1
            self.event_labels =  [event['name'] for event in events_format]

        self.events_discard_format = events_discard_format[:]
        self.events_discard_format += [
            {
                'name': sf['h5_path'] + '_mask',
                'h5_path': sf['h5_path'] + '_mask',
                'extend': 0
            } for sf in signals_format
        ]

        if not transformations:
            self.transformations = {}
        else:
            self.transformations = {
                regu: regu_function(transformations_factor) for regu, regu_function in regularizers.items() if regu in transformations
            }

        get_data = get_h5_data
        get_events = get_h5_events
        if cache_data:
            memory = Memory(h5_directory + "/.cache/", mmap_mode="r", verbose=0)
            get_data = memory.cache(get_h5_data)
            get_events = memory.cache(get_h5_events)

        # window parameters
        self.records = records
        self.window = window
        self.fs = fs 
        self.window_size = int(window * fs)
        self.number_of_channels = number_of_channels
        self.prediction_resolution = prediction_resolution
        self.overlap = overlap
        self.batch_size = batch_size
        self.events_format = events_format
        self.discard_threshold = discard_threshold
        self.select_threshold = select_threshold
        self.minimum_overlap = minimum_overlap
        self.signals_format = signals_format
        self.batch_normalization = batch_normalization
        self.h5_directory = h5_directory
        self.predictions_per_window = window // prediction_resolution

        # open signals and events
        self.signals = {}
        self.events = {}
        self.events_discard = {}
        self.events_select = {}
        self.index_to_record = []
        self.index_to_record_event = {e['name']: [] for e in events_format}

        data = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_data)(
            filename="{}/{}".format(h5_directory, record),
            signals=signals_format
        ) for record in self.records)

        for record, data in zip(self.records, data):
            signal_size = data.shape[0]
            number_of_windows = max(1, int((signal_size - self.window_size) // (self.window_size * (1 - self.overlap)) + 1))
            self.signals[record] = {
                "data": data,
                "size": signal_size,
            }

            self.index_to_record.extend([
                {
                    "record": record,
                    "max_index": max(0, signal_size - self.window_size),
                    "index": int(x * self.window_size * (1 - self.overlap))
                } for x in range(number_of_windows)
            ])

            self.events[record] = {}
            self.events_discard[record] = {}
            self.events_select[record] = {}

            # Discard events
            for label, event in enumerate(self.events_discard_format):

                data = get_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                    fs=self.fs,
                )

                self.events_discard[record][event["name"]] = {
                    "data": data,
                    "label": label,
                }

            # Select events
            for label, event in enumerate(events_select_format):
                data = get_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                    fs=self.fs,
                )

                self.events_select[record][event["name"]] = {
                    "data": data,
                    "label": label,
                }

            for label, event in enumerate(events_format):

                data = get_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                    fs=self.fs,
                )

                for ed_key, ed_it in self.events_discard[record].items():
                    overlap = intersection_overlap(np.array( [ed_it['data'][0, :], ed_it['data'][0, :] + ed_it['data'][1, :]]).T,
                                                   np.array([data[0, :], data[0, :] + data[1, :]]).T)
                    if len(overlap) > 0:
                        max_iou = overlap.max(0)
                        keep_bool = (max_iou < (self.discard_threshold * self.fs))
                        data = data[:, keep_bool]

                for es_key, es_it in self.events_select[record].items():
                    overlap = intersection_overlap(np.array( [es_it['data'][0, :], es_it['data'][0, :] + es_it['data'][1, :]]).T,
                                                   np.array([data[0, :], data[0, :] + data[1, :]]).T)
                    if len(overlap) > 0:
                        max_iou = overlap.max(0)
                        keep_bool = (max_iou > (self.select_threshold * self.fs))
                        data = data[:, keep_bool]

                self.events[record][event["name"]] = {
                    "data": data,
                    "label": label,
                }

                self.index_to_record_event[event["name"]].extend([{
                    "record": record,
                    "max_index": max(0, signal_size - self.window_size),
                    "data": data[:, n]
                } for n in range(data.shape[1])])


        if val_rate > 0:
            random.seed(seed)
            #for label, event in enumerate(events_format):
            #    index_val = int(len(self.index_to_record_event[event["name"]]) * val_rate)
            #    random.shuffle(self.index_to_record_event[event["name"]]) # maybe dont do this
            #    if dataset_type == 'validation':
            #        self.index_to_record_event[event["name"]] = self.index_to_record_event[event["name"]][:index_val]
            #    elif dataset_type == 'train':
            #        self.index_to_record_event[event["name"]] = self.index_to_record_event[event["name"]][index_val:]
            index_val = int(len(self.index_to_record) * val_rate)
            random.shuffle(self.index_to_record)
            if dataset_type == 'validation':
                self.index_to_record = self.index_to_record[:index_val]
            elif dataset_type == 'train':
                self.index_to_record = self.index_to_record[index_val:]

    def __len__(self):
        return len(self.index_to_record) // self.batch_size + 1
    
    def __getitem__(self, idx):
        idx = idx % int(len(self.index_to_record) // self.batch_size)
        signal_batch, events_batch, mask_batch = [], [], []
        for idx_ in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            idx_ = idx_ % len(self.index_to_record)
            signal, events = self.get_sample(
                record=self.index_to_record[idx_]['record'],
                index=self.index_to_record[idx_]['index'])
            if self.batch_normalization:
                signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
            signal_batch += [signal]
            events_batch += [events]

        return np.array(signal_batch).astype('float32'), np.array(events_batch).astype('float32')


    def get_sample(self, record, index):

        signal_data = self.signals[record]["data"][index: index + self.window_size, :]
        events = self.get_events_sample(record=record, index=index, events=self.events[record], minimum_overlap=self.minimum_overlap)
        masks = self.get_events_sample(record=record, index=index, events=self.events_discard[record], minimum_overlap=0, any_overlap=True)

        # apply mask:
        mask_any = (np.sum(masks, axis=-1) > 0)
        events[mask_any, :] = -1

        return signal_data, events

    def get_events_sample(self, record, index, events, minimum_overlap, any_overlap=False, minimum_duration=10):
        """Return a sample [data, events] from a record at a particular index"""

        events_semantic = np.zeros((self.predictions_per_window, len(events)))
        count = 0
        for event_name, event in  events.items():

            starts, durations = event["data"][0, :], event["data"][1, :]

            events_data = []

            if len(starts) > 0:

                # Relative start stop
                starts_relative = (starts - index) / self.window_size
                durations_relative = durations / self.window_size
                stops_relative = starts_relative + durations_relative

                if any_overlap:
                    starts_relative[stops_relative > 0] = np.maximum(starts_relative[stops_relative > 0], 0)
                    durations_relative = durations_relative
                    stops_relative[starts_relative<1] = np.minimum(stops_relative[starts_relative<1], 1)

                # Find valid start or stop
                valid_starts_index = np.where((starts_relative >= 0) * (starts_relative < 1))[0]
                valid_stops_index = np.where((stops_relative > 0) * (stops_relative <= 1))[0]

                # merge them
                valid_indexes = set(list(valid_starts_index) +
                                    list(valid_stops_index))
                # TODO - the mask should not be subject to the minimum overlap requirement.

                # Annotations contains valid index with minimum overlap requirement
                for valid_index in valid_indexes:
                    if (durations[valid_index] / self.fs) > minimum_duration:
                        if (valid_index in valid_starts_index) \
                                and (valid_index in valid_stops_index):
                            events_data.append((float(starts_relative[valid_index]),
                                                float(stops_relative[valid_index])))
                        elif valid_index in valid_starts_index:
                            if ((1 - starts_relative[valid_index]) / durations_relative[valid_index]) > minimum_overlap:
                                events_data.append((float(starts_relative[valid_index]), 1))
                        elif valid_index in valid_stops_index:
                            if ((stops_relative[valid_index]) / durations_relative[valid_index]) > minimum_overlap:
                                events_data.append((0, float(stops_relative[valid_index])))
            events_semantic[:, count] = semantic_formating(output_size=self.predictions_per_window,
                                                           events=events_data,
                                                           sufficient_overlap=7.5/self.prediction_resolution)

            count += 1

        return events_semantic

    def get_record_events(self, record):

        events = [[] for _ in range(self.number_of_classes)]

        for event_data in self.events[record].values():
            events[event_data["label"]].extend([
                [start // self.fs, (start + duration) // self.fs]
                for start, duration in event_data["data"].transpose().tolist()
            ])

        return events


    def get_record_batch(self, record):

        record_indexes = [index for index in self.index_to_record if index['record'] == record]

        signals, start_times = [], []
        for record_index in record_indexes:

            signal = self.signals[record]["data"][record_index['index']: record_index['index'] + self.window_size, :]
            if self.batch_normalization:
                signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
            signals += [signal]
            start_times += [record_index['index'] // self.fs]

            if len(signals) == self.batch_size:
                yield np.array(signals), np.array(start_times)
                signals, masks, start_times = [], [], []
        if signals:
            yield np.array(signals), np.array(start_times)

    def get_record_events_by_label(self, record, event_label, extent_event_window=0):

        data = get_h5_events(
            filename="{}/{}".format(self.h5_directory, record),
            event=event_label,
            fs=self.fs,
        )

        data_ = np.array(data)
        data_[0, :] += extent_event_window
        data_[1, :] += 2 * extent_event_window

        return data_

    def get_TST(self, record):
        hyp_events = self.get_record_events_by_label(record, event_label={'h5_path': 'sleep'})
        return (hyp_events[1,:]).sum() // self.fs

    def get_hypnogram_semantic(self, record):

        WAKE = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'wake'}) // self.fs, 0, -1).tolist()
        NREM = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'nrem'}) // self.fs, 0, -1).tolist()
        REM = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'rem'}) // self.fs, 0, -1).tolist()
        WAKE  = [[w[0], w[0] + w[1]] for w in WAKE]
        NREM = [[n[0], n[0] + n[1]] for n in NREM]
        REM = [[r[0], r[0] + r[1]] for r in REM]

        signal_size = (self.signals[record]['data']).shape[0] // self.fs
        hypnogram = np.zeros((signal_size, 3))
        hypnogram[:, 0] = semantic_formating(signal_size, WAKE)
        hypnogram[:, 1] = semantic_formating(signal_size, NREM)
        hypnogram[:, 2] = semantic_formating(signal_size, REM)
        k = 1
        return hypnogram


    def compile_event(self, event_label='', reference_point='start', model=None, plot_individual=False, name=''):

        signals_compiled = []
        events_compiled = []
        predictions = []
        prediction = None

        for event_num, event_inst in enumerate(self.index_to_record_event[event_label]):
            if reference_point == 'start':
                index = event_inst['data'][0] - (self.window_size / 2)
            else:
                index = (event_inst['data'][0] + event_inst['data'][1]) - (self.window_size / 2)

            if index > 0 and index < event_inst['max_index']:
                signal, event = self.get_sample(record=event_inst['record'], index=int(index))
                if self.batch_normalization:
                    signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
                if model is not None:
                    prediction = model.predict(np.array([signal]))[0, :, :]
                    predictions += [prediction]
                if plot_individual:
                    if len(self.input_shape) == 2:
                        self.plot(name=str(event_num) + '_' + name,
                                  path_ext=event_inst['record'] + '_' + event_label,
                                  signals=signal,
                                  targets=event,
                                  predictions=prediction)
                    else:
                        self.plot(str(event_num) + '_' + name,
                                  path_ext=event_inst['record'] + '_' + event_label,
                                  spectrograms=signal,
                                  targets=event,
                                  predictions=prediction)

                signals_compiled += [signal]
                events_compiled += [event]

        return np.array(signals_compiled), np.array(events_compiled), np.array(predictions)

    def compile_windows(self, event_label='', reference_point='start', model=None, plot_individual=False, name=''):

        signals_compiled = []
        events_compiled = []
        predictions = []
        prediction = None

        for event_num, event_inst in enumerate(self.index_to_record):
            signal, event, mask = self.get_sample(record=event_inst['record'], index=event_inst['index'])
            if self.batch_normalization:
                signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
            if model is not None:
                prediction = model.predict(np.array([signal]))[0, :, :]
                predictions += [prediction]
            if plot_individual:
                if len(self.input_shape) == 2:
                    self.plot(name=name + ' ' + str(event_num),
                              path_ext=event_inst['record'],
                              signals=signal,
                              targets=event,
                              predictions=prediction)
                else:
                    self.plot(name=str(event_num),
                              path_ext=event_inst['record'],
                              spectrograms=signal,
                              targets=event,
                              predictions=prediction)

            signals_compiled += [signal]
            events_compiled += [event]

        return np.array(signals_compiled), np.array(events_compiled), np.array(predictions)

    def compile_record(self, record, event_label='', reference_point='start', model=None, plot_individual=False, name=''):

        assert self.overlap == 0

        signals_compiled = []
        events_compiled = []
        mask_compiled = []
        predictions = []
        prediction = None

        for event_num, event_inst in enumerate(self.index_to_record):
            if event_inst['record'] == record:
                signal, event = self.get_sample(record=event_inst['record'], index=event_inst['index'])
                if self.batch_normalization:
                    signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
                if model is not None:
                    prediction = model.predict(np.array([signal]))[0, :, :]
                    predictions += [prediction]
                if plot_individual:
                    if len(self.input_shape) == 2:
                        self.plot(name=name + ' ' + str(event_num),
                                  path_ext=event_inst['record'],
                                  signals=signal,
                                  targets=event,
                                  predictions=prediction)
                    else:
                        self.plot(name=str(event_num),
                                  path_ext=event_inst['record'],
                                  spectrograms=signal,
                                  targets=event,
                                  predictions=prediction)

                signals_compiled += [signal]
                events_compiled += [event]

        # reformatting
        signals = np.array(signals_compiled)
        events = np.array(events_compiled)
        predictions = np.array(predictions)
        signals = np.reshape(signals, (signals.shape[0] * signals.shape[1], signals.shape[2]))
        events = np.reshape(events, (events.shape[0] * events.shape[1], events.shape[2]))
        predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1], predictions.shape[2]))
        hypnogram = self.get_hypnogram_semantic(record=record)

        return signals, events, predictions, hypnogram


    def plot(self, name='', path='', signals=None, spectrograms=None, targets=None, predictions=None, hypnogram=None, masks=None):

        num_subplot_rows = 0
        if signals is not None: num_subplot_rows += len(self.signals_format)
        if spectrograms is not None: num_subplot_rows += (spectrograms.shape[-1] * 1)
        if targets is not None: num_subplot_rows += 1
        if predictions is not None: num_subplot_rows += 1
        if hypnogram is not None: num_subplot_rows += 1
        if masks is not None: num_subplot_rows += 1

        subplot_count = 0

        if num_subplot_rows > 0:

            # PLOT
            gs = gridspec.GridSpec(num_subplot_rows, 1)
            fig = plt.figure(figsize=(60, 2 * num_subplot_rows), dpi=300)
            fig.suptitle(name, fontsize=14)

            if signals is not None:
                ts_signal = np.arange(start=0, stop=(signals.shape[0]), step=1) / self.fs
                num1 = [[0], [1, 2, 3, 4], [5]]
                num2 = [[[0, 1, 2, 3]]]
                for num, signal_format in enumerate(self.signals_format):
                    ax = fig.add_subplot(gs[subplot_count])
                    ax.plot(ts_signal, signals[:, num1[num]],
                            label=signal_format['h5_path'],
                            linewidth=.4)
                    ax.grid(which='both')
                    ax.title.set_text(signal_format['h5_path'])
                    ax.set_ylabel(r'amplitude', size=12)
                    ax.set_xlabel(r'time (s)', size=12)
                    ax.legend()
                    subplot_count += 1
                name += '_raw'

            if spectrograms is not None:
                ts_signal = np.arange(start=0, stop=(spectrograms.shape[0]), step=1) / self.fs
                for num, signal_format in enumerate(self.signals_format):
                    # frequency range
                    freq_range = [(preproc['args']['f_min'], preproc['args']['f_max']) for preproc in
                                  signal_format['preprocessing'] if preproc['type'] == 'psd'][0]

                    ax = fig.add_subplot(gs[subplot_count])
                    S = np.swapaxes(spectrograms[:, :, num], axis1=1, axis2=0)
                    S = np.flipud(S)
                    ax.imshow(S, extent=(np.amin(ts_signal), np.amax(ts_signal), freq_range[0], freq_range[1]),
                              cmap=cm.hot, aspect='auto') #, vmax=abs(S).max(), vmin=-abs(S).max())
                    ax.title.set_text(signal_format['h5_path'])
                    ax.set_ylabel(r'freq (Hz)', size=12)
                    ax.set_xlabel(r'time (s)', size=12)
                    subplot_count += 1
                name += '_spec'

            if targets is not None:
                ts_event = np.arange(start=0, stop=(targets.shape[0])* self.prediction_resolution, step=self.prediction_resolution)
                ax = fig.add_subplot(gs[subplot_count], sharex=ax)
                for event_num, event in enumerate(self.events_format):
                    ax.plot(ts_event, targets[:, event_num],
                            '-',
                            label=event['name'] + '_target',
                            color="C{}".format(event_num),
                            linewidth=1)
                ax.title.set_text(r'targets')
                ax.set_xlabel(r'time (s)', size=12)
                ax.set_ylim([0, 1.1])
                ax.legend()
                ax.grid(which='both')
                subplot_count += 1

            if predictions is not None:
                ts_event = np.arange(start=0, stop=(predictions.shape[0]) * self.prediction_resolution,
                                     step=self.prediction_resolution)
                ax = fig.add_subplot(gs[subplot_count], sharex=ax)
                #for event_num, event in enumerate(self.events_format):
                for event_num, event in enumerate([{'name': 'SDB'}]):
                    ax.plot(ts_event, predictions[:, event_num],
                            '-',
                            label=event['name'] + '_target',
                            color="C{}".format(event_num),
                            linewidth=1)
                ax.title.set_text(r'predictions')
                ax.set_xlabel(r'time (s)', size=12)
                ax.set_ylim([0, 1.1])
                ax.legend()
                ax.grid(which='both')
                subplot_count += 1

            if hypnogram is not None:
                ts_hyp = np.arange(start=0, stop=(hypnogram.shape[0]), step=1)
                ax = fig.add_subplot(gs[subplot_count], sharex=ax)
                ax.plot(ts_hyp, np.matmul(hypnogram, np.array([[1], [2], [3]])),
                        linewidth=1)
                ax.title.set_text(r'hypnogram')
                ax.set_xlabel(r'time (s)', size=12)
                ax.set_ylim([0, 3.1])
                ax.grid(which='both')
                plt.yticks(range(4), ['none', 'wake', 'nrem', 'rem'])
                subplot_count += 1

            if not os.path.isdir(path):
                os.mkdir(os.path.join(path))
            plt.savefig(os.path.join(path, name + '.png'))
        else:
            print('nothing to plot')
