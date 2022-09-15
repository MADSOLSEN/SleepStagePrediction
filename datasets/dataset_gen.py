import numpy as np
from tensorflow.keras.utils import Sequence
from joblib import Memory, Parallel, delayed
from utils import check_inf_nan
import random
import tqdm
np.seterr(all='raise')

from utils import get_h5_data, get_h5_events, semantic_formating, get_h5_auxiliary
from signal_processing import normalizers, regularizers


class DatasetGenerator(Sequence):

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
        Down-sampling rate to apply to signals
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
                 number_of_channels,
                 events_format,
                 events_discard_format=[],
                 prediction_resolution=1,
                 overlap=0.5,
                 minimum_overlap=0.1,
                 batch_size=64,
                 mode='inference',
                 cache_data=True,
                 n_jobs=1,
                 use_mask=True,
                 load_signal_in_RAM=True):

        # datasets
        self.records = records
        self.h5_directory = h5_directory

        # signal modalities
        self.signals_format = signals_format
        self.window = window
        self.number_of_channels = number_of_channels
        self.prediction_resolution = prediction_resolution
        self.overlap = overlap
        self.batch_size = batch_size
        self.predictions_per_window = window // prediction_resolution
        self.nChannels = sum([sf['dimensions'][-1] for sf in signals_format.values()])
        self.nSpace = [sf['dimensions'][0] for sf in signals_format.values()][0] # assumes same space resolution
        self.fsTime = [sf['fs_post'] for sf in signals_format.values()][0] # assumes same temporal resolution

        # events
        self.events_format = events_format
        self.minimum_overlap = minimum_overlap
        self.number_of_events = len(events_format)
        self.number_of_classes = len(events_format)
        self.event_probabilities = [event['probability'] for event in events_format]
        self.event_labels = [event['name'] for event in events_format]
        assert sum(self.event_probabilities) <= 1

        # training
        self.mode = mode
        self.load_signal_in_RAM = load_signal_in_RAM
        self.use_mask = use_mask
        self.transformations = regularizers

        # Preallocation
        self.signals = {}
        self.events = {}
        self.events_discard = {}
        self.events_select = {}
        self.index_to_record = []
        self.index_to_record_event = {e['name']: [] for e in events_format}

        if use_mask:
            self.events_discard_format = events_discard_format[:]
            self.events_discard_format += [
                {
                    'name': sf['h5_path'] + '_mask',
                    'h5_path': sf['h5_path'] + '_mask',
                    'extend': 0
                } for signal_name, sf in signals_format.items() # if not in  ['ppg_features', 'PPG_features', 'RR', 'RR_detrend']
            ]
        else:
            self.events_discard_format = []


        self.load_data = get_h5_data
        self.load_events = get_h5_events

        if cache_data:
            memory = Memory(h5_directory + "/.cache/", mmap_mode="r", verbose=0)
            self.load_data = memory.cache(get_h5_data)
            self.load_events = memory.cache(get_h5_events)

        data = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(self.load_data)(
            filename="{}/{}".format(h5_directory, record),
            signals=signals_format
        ) for record in tqdm.tqdm(records))


        for record, data in zip(records, data):

            signal_durations = [ int(data[signal_name].shape[0] / signal_format['fs_post']) for signal_name, signal_format in self.signals_format.items()]
            record_duration = min(signal_durations)
            #record_duration_clipped = record_duration - (record_duration % prediction_resolution) - 1
            record_duration_clipped = record_duration - (record_duration % 30)
            number_of_windows = max(1, int((record_duration_clipped - self.window) // (self.window * (1 - self.overlap)) + 1))


            # find signal info:
            if self.load_signal_in_RAM:
                for key, item in data.items():
                    data[key] = np.array(item).astype('float32')
                self.signals[record] = {"data": data}

            self.index_to_record.extend([
                {
                    "record": record,
                    "max_index": int(max(0, record_duration_clipped - self.window)),
                    "index": int(x * self.window * (1 - self.overlap))
                } for x in range(number_of_windows)] + [
                {
                    "record": record,
                    "max_index": int(max(0, record_duration_clipped - self.window)),
                    "index": int(max(0, record_duration_clipped - self.window))
                }
            ])

            self.events[record] = {}
            self.events_discard[record] = {}
            self.events_select[record] = {}

            # Discard events
            for label, event in enumerate(self.events_discard_format):
                data = self.load_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                )
                self.events_discard[record][event["name"]] = {
                    "data": np.array(data).astype('float32'),
                    "label": label,
                }

            for label, event in enumerate(events_format):

                events = self.load_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                )

                self.events[record][event["name"]] = {
                    "data": np.array(events).astype('float32'),
                    "label": label,
                }

                self.index_to_record_event[event["name"]].extend([{
                    "record": record,
                    "max_index": max(0, record_duration_clipped - self.window),
                    "data": np.array(events[:, n]).astype('float32')
                } for n in range(events.shape[1])])


    def __len__(self):
        max_len = 10000
        return min(len(self.index_to_record) // (self.batch_size), max_len)

    def __getitem__(self, idx):

        signal_batch = np.zeros((self.batch_size, int(self.window * self.fsTime), self.nSpace, self.nChannels)).astype('float32')
        event_batch = []

        for num, idx_ in enumerate(range(idx * self.batch_size, (idx + 1) * self.batch_size)):

            # signals:
            signals = []
            for signal_name, signal_format in self.signals_format.items():

                signal = self.get_signals(record=self.index_to_record[idx_]['record'],
                                          signal_name=signal_name,
                                          index=self.index_to_record[idx_]['index'])
                if signal_format['batch_normalization']:
                    signal = normalizers[signal_format['batch_normalization']['type']](signal, **signal_format['batch_normalization']['args'])
                if self.mode == 'train': # data augmentation:
                    for trans, item in signal_format['transformations'].items():
                        signal = self.transformations[trans](signal, **item)
                signals += [signal]

            signal_batch[num, :signal.shape[0], :, :] = np.stack(signals, axis=2)

            # events:
            event_batch += [self.get_events(record=self.index_to_record[idx_]['record'], index=self.index_to_record[idx_]['index'])]

        return signal_batch, np.stack(event_batch, axis=0).astype('float32')

    def get_record_item(self, record):

        signals = []

        for signal_name, signal_format in self.signals_format.items():
            signal = self.get_signals_by_record(record=record, signal_name=signal_name)
            if signal_format['batch_normalization']:
                signal = normalizers[signal_format['batch_normalization']['type']](signal, **signal_format['batch_normalization']['args'])
            if self.mode == 'train':
                for trans, item in signal_format['transformations'].items():
                    signal = self.transformations[trans](signal, **item)
            signals += [signal]

        return np.array(signals).transpose(1, 2, 0)

    def get_model_input_by_record(self, record):

        signal = self.get_record_item(record=record)

        if len(self.h5_directory_auxiliary) > 0:
            auxiliary = self.get_entire_auxiliary_signal(record=record)
            return [signal, auxiliary]
        return signal



    def get_signals_by_record(self, record, signal_name):
        if self.load_signal_in_RAM:
            signal_data = self.signals[record]["data"][signal_name]
        else:
            data = self.load_data(filename="{}/{}".format(self.h5_directory, record), signals=self.signals_format)
            signal_data = data[signal_name]
        return signal_data

    def get_auxiliary_item(self, index_to_record):
        auxiliary_batch = []
        for counter, itr in enumerate(index_to_record):
            auxiliary_batch += [self.get_auxiliary_signal(record=itr['record'], index=itr['index'])]
        return np.array(auxiliary_batch).astype('float32')


    def signal_model_prep(self, signal_batch):
        model_input = []
        if isinstance(self.model_format, dict):
            sig = np.concatenate([signal_batch[signal_name] for signal_name in self.model_format['signals']], axis=-1)
            check_inf_nan(sig)
            model_input = sig # += [sig]
        else:
            for models in self.model_format:
                model_input += [np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)]
        return model_input

    def get_signals(self, record, signal_name, index):
        fs = self.signals_format[signal_name]['fs_post']
        if self.load_signal_in_RAM:
            signal_data = self.signals[record]["data"][signal_name][int(index * fs): int((index + self.window) * fs), :]
        else:
            data = self.load_data(filename="{}/{}".format(self.h5_directory, record), signals=self.signals_format)
            signal_data = data[signal_name][int(index * fs): int((index + self.window) * fs), :]

        # assert(signal_data.shape[0] == len(list(range(int(index * fs), int((index + self.window) * fs))))) # If this fails - you need to redefine max_index
        return signal_data

    def get_events(self, record, index):
        events = self.get_events_sample(index=index, events=self.events[record], minimum_overlap=self.minimum_overlap)
        if self.use_mask:
            #masks = self.get_events_sample(index=index, events=self.events_discard[record], minimum_overlap=0, any_overlap=True)
            #mask_any = (np.sum(masks, axis=-1) > 0)
            #events[mask_any, :] = -1

            masks = self.get_events_sample(index=index, events=self.events_discard[record], minimum_overlap=0, any_overlap=True)
            mask_any = (np.sum(masks, axis=-1) > 0)
            #mask_non_event = (np.sum(events, axis=-1) == 0)  # mask if no event
            events[mask_any, :] = -1
            #events[mask_non_event, :] = -1

        return events

    def get_auxiliary_signal(self, record, index):
        data, fs = get_h5_auxiliary(filename="{}\\{}\\{}".format(self.h5_directory, self.h5_directory_auxiliary, record),
                                    labels=['wake', 'light', 'deep', 'rem'])
        x_aux = data[int(index * fs): int((index + self.window) * fs)]
        return x_aux



    def get_entire_auxiliary_signal(self, record, desired_fs=1):
        data, fs = get_h5_auxiliary(filename="{}\\{}\\{}".format(self.h5_directory, self.h5_directory_auxiliary, record),
                                    labels=['wake', 'light', 'deep', 'rem'])
        if desired_fs > fs: # higher resolution requested
            data = np.repeat(data, int(desired_fs/fs), axis=0)
        elif desired_fs < fs:
            data = data[::int(fs / desired_fs), :]

        return data

    def get_events_sample(self, index, events, minimum_overlap, any_overlap=True, minimum_duration=10):
        """Return a sample [data, events] from a record at a particular index"""
        events_semantic = np.zeros((self.predictions_per_window, len(events)))
        count = 0
        for event_name, event in  events.items():

            starts, durations = event["data"][0, :], event["data"][1, :]
            events_data = []

            if len(starts) > 0:

                # Relative start stop
                starts_relative = (starts - index) / self.window
                durations_relative = durations / self.window
                stops_relative = starts_relative + durations_relative

                if any_overlap:
                    starts_relative[stops_relative > 0] = np.maximum(starts_relative[stops_relative > 0], 0)
                    durations_relative = durations_relative
                    stops_relative[starts_relative<1] = np.minimum(stops_relative[starts_relative<1], 1)

                # valid if event is contained inside window # Therefore any overlap is used for masks!
                valid_starts_index = np.where((starts_relative >= 0) * (starts_relative < 1))[0]
                valid_stops_index = np.where((stops_relative > 0) * (stops_relative <= 1))[0]

                # merge them
                valid_indexes = set(list(valid_starts_index) +
                                    list(valid_stops_index))
                # TODO - the mask should not be subject to the minimum overlap requirement.

                # Annotations contains valid index with minimum overlap requirement
                for valid_index in valid_indexes:
                    if (durations[valid_index]) > minimum_duration:
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
                [start, (start + duration)]
                for start, duration in event_data["data"].transpose().tolist()
            ])

        return events

    def get_record_batch(self, record):

        record_indexes = [index for index in self.index_to_record if index['record'] == record]
        start_times = [start_time['index'] for start_time in record_indexes]
        signals, events = self.get_item(index_to_record=record_indexes)
        model_input = self.signal_model_prep(signal_batch=signals)
        if len(self.h5_directory_auxiliary) > 0:
            auxiliary = self.get_auxiliary_item(index_to_record=record_indexes)
            yield [model_input, auxiliary], start_times
        else:
            yield [model_input], np.array(start_times)

    def get_record_input(self, record):
        record_indexes = [index for index in self.index_to_record if index['record'] == record]



    def hypnogram_plot_helper(self, x, mask=None):
        """

        :param x: vector of sleep stages [T x 4]. Assumed order: ['wake', 'light', 'deep', 'rem'].
        :return: sleep stage vector with order that corresponds to traditional visualization: [T x 1].
        ['wake', 'rem', 'light', 'deep'].

        """

        x_reordered = np.stack([x[:, 2], x[:, 1], x[:, 3], x[:, 0]], axis=1)  # reorder to traditional view
        x_argmax = np.argmax(x_reordered, axis=1)  # take argmax
        x_argmax[mask] = -1
        return x_argmax


    def get_record_batch_(self, record):

        record_indexes = [index for index in self.index_to_record if index['record'] == record]
        signal_batch = {signal_name: np.zeros([self.batch_size,
                                               int(self.window * signal_format['fs_post'])] + signal_format['dimensions'])
            for signal_name, signal_format in self.signals_format.items()}

        counter, start_times = 0, []
        for record_index in record_indexes:
            for signal_name, signal_format in self.signals_format.items():
                signal = self.get_signals(
                    record=record_index['record'],
                    signal_name=signal_name,
                    index=record_index['index'])
                if signal_format['batch_normalization']:
                    signal = normalizers[signal_format['batch_normalization']['type']](signal, **signal_format['batch_normalization']['args'])
                if self.mode == 'train':
                    for trans in signal_format['transformations'].keys():
                        signal = self.transformations[trans](signal)

                signal_batch[signal_name][counter, :signal.shape[0], :] = np.reshape(signal, [signal.shape[0]] + signal_format['dimensions'])
            start_times += [record_index['index']]
            counter += 1
            if counter == self.batch_size:
                model_input = []
                if isinstance(self.model_format, dict):
                    sig = np.concatenate([signal_batch[signal_name] for signal_name in self.model_format['signals']],
                                         axis=-1)
                    check_inf_nan(sig)
                    model_input += [sig]
                else:
                    for models in self.model_format:
                        sig = np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)
                        check_inf_nan(sig)
                        model_input += [sig]
                yield model_input, np.array(start_times)
                signal_batch = {
                    signal_name: np.zeros(
                        [self.batch_size, int(self.window * signal_format['fs_post'])] +
                        signal_format['dimensions'])
                    for signal_name, signal_format in self.signals_format.items()}
                counter, start_times = 0, []


        if signal_batch:
            model_input = []
            if isinstance(self.model_format, dict):
                sig = np.concatenate([signal_batch[signal_name] for signal_name in self.model_format['signals']],
                                     axis=-1)
                check_inf_nan(sig)
                model_input += [sig]
            else:
                for models in self.model_format:
                    sig = np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)
                    check_inf_nan(sig)
                    model_input += [sig]
            yield model_input, np.array(start_times)

    def get_record_events_by_label(self, record, event_label, extent_back=0, extent_forward=0):

        data = get_h5_events(
            filename="{}/{}".format(self.h5_directory, record),
            event=event_label
        )

        data_ = np.array(data)
        data_[0, :] -= extent_back
        data_[1, :] += extent_back + extent_forward

        return data_

    def get_number_of_events(self, record, event_label):
        try:
            events_ = self.get_record_events_by_label(record=record, event_label={'h5_path': event_label})
            out = 0 if len(events_) == 0 else events_.shape[1]
        except:
            out = 0
        return out


    def get_TST(self, record):
        hyp_events = self.get_record_events_by_label(record, event_label={'h5_path': 'sleep'})
        return (hyp_events[1,:]).sum()


    def get_rec_dur(self, record):
        data = self.load_data(filename="{}/{}".format(self.h5_directory, record), signals=self.signals_format)
        num_samples = next(iter(data.values())).shape[0]
        sample_frequency = next(iter(self.signals_format.values()))['fs_post']
        return int(num_samples / sample_frequency)



class BalancedDatasetGenerator(DatasetGenerator):
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
                 events_discard_format=[],
                 events_format=None,
                 prediction_resolution=1,
                 overlap=0.5,
                 minimum_overlap=0.5,
                 batch_size=64,
                 seed=2022,
                 mode='inference',
                 use_mask=True,
                 cache_data=True,
                 load_signal_in_RAM=True):

        super().__init__(records=records,
                         h5_directory=h5_directory,
                         signals_format=signals_format,
                         window=window,
                         events_discard_format=events_discard_format,
                         number_of_channels=number_of_channels,
                         events_format=events_format,
                         overlap=overlap,
                         prediction_resolution=prediction_resolution,
                         minimum_overlap=minimum_overlap,
                         batch_size=batch_size,
                         mode=mode,
                         use_mask=use_mask,
                         cache_data=cache_data,
                         load_signal_in_RAM=load_signal_in_RAM)

        self.np_random = np.random.seed(seed)
        self.random = random.seed(seed)

    def __getitem__(self, idx):
        signal_batch, events_batch = [], []
        while len(signal_batch) < self.batch_size:
            signal, events = self.extract_balanced_data()
            if self.batch_normalization:
                signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
            if self.mode == 'train':
                for trans in self.transformations.keys():  # augmentation
                    signal = self.transformations[trans](signal)
            signal_batch += [signal]
            events_batch += [events]
        return np.array(signal_batch).astype('float32'), np.array(events_batch).astype('float32')


    def extract_balanced_data(self):

        event = self.random.choice((self.number_of_classes + 1),
                                 p=self.event_probabilities + [1 - sum(self.event_probabilities)])

        if event == (self.number_of_classes):
            # get random segment from file
            random_event = self.random.sample(self.index_to_record, 1)[0]
            index = np.random.randint(random_event['max_index'])
            signal_data, events_data = self.get_sample(random_event['record'], int(index))
            while np.sum(events_data) != 0:
                random_event = self.random.sample(self.index_to_record, 1)[0]
                index = self.np_random.randint(random_event['max_index'])
                signal_data, events_data = self.get_sample(random_event['record'], int(index))
        else:
            # get random event from random file.
            random_event = self.random.sample(self.index_to_record_event[self.events_format[event]['name']], 1)[0]
            try:
                if (self.window <= random_event["data"][1]) or random_event["data"][0] == 0:
                    index = random_event["data"][0]
                elif random_event["data"][0] >= random_event['max_index']:
                    index = random_event['max_index']
                else:
                    index = self.np_random.randint(
                        low=max(0, random_event["data"][0] - (self.window - random_event["data"][1])),
                        high=min(random_event['max_index'], random_event["data"][0]))
            except:
                print(random_event)
            signal_data, events_data = self.get_sample(random_event['record'], int(index))

        return signal_data, events_data


