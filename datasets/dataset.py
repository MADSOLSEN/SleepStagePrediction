import os
import sys
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import Sequence
from joblib import Memory, Parallel, delayed
from utils import check_inf_nan
import random
import seaborn as sns
from copy import deepcopy
import pywt
np.seterr(all='raise')

from utils import get_h5_data, get_h5_events, semantic_formating, intersection_overlap, jaccard_overlap, any_formating, create_directory_tree, binary_to_array
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
                 model_format=None,
                 events_format=None,
                 events_discard_format=[],
                 events_select_format=[],
                 prediction_resolution=1,
                 overlap=0.5,
                 minimum_overlap=0.5,
                 batch_size=64,
                 mode='inference',
                 transformations_factor=0.1,
                 discard_threshold=10,
                 select_threshold=10,
                 cache_data=True,
                 n_jobs=1,
                 val_rate=0,
                 dataset_type='',
                 seed=2020,
                 use_mask=True,
                 load_signal_in_RAM=True):
        # load_signal_in_RAM = False
        # cache_data = False
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
            } for signal_name, sf in signals_format.items() # if not in  ['ppg_features', 'PPG_features', 'RR', 'RR_detrend']
        ]

        self.events_select_format = []

        #self.events_select_format = [
        #    {
        #        'name': 'sleep',
        #        'h5_path': 'sleep',
        #        'extend': 0
        #    }
        #]

        print(self.events_discard_format)

        self.load_data = get_h5_data
        self.load_events = get_h5_events
        #cache_data = False
        if cache_data:
            memory = Memory(h5_directory + "/.cache/", mmap_mode="r", verbose=0)
            self.load_data = memory.cache(get_h5_data)
            self.load_events = memory.cache(get_h5_events)

        # window parameters
        self.mode = mode
        self.transformations = regularizers
        self.model_format = model_format
        self.load_signal_in_RAM = load_signal_in_RAM
        self.use_mask = use_mask
        self.records = records
        self.window = window
        self.number_of_channels = number_of_channels
        self.prediction_resolution = prediction_resolution
        self.overlap = overlap
        self.batch_size = batch_size
        self.events_format = events_format
        self.discard_threshold = discard_threshold
        self.select_threshold = select_threshold
        self.minimum_overlap = minimum_overlap
        self.signals_format = signals_format
        self.h5_directory = h5_directory
        self.predictions_per_window = window // prediction_resolution

        # open signals and events
        self.signals = {}
        self.events = {}
        self.events_discard = {}
        self.events_select = {}
        self.index_to_record = []
        self.index_to_record_event = {e['name']: [] for e in events_format}


        data = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(self.load_data)(
            filename="{}/{}".format(h5_directory, record),
            signals=signals_format
        ) for record in records)


        for record, data in zip(records, data):

            signal_durations = [ int(data[signal_name].shape[0] / signal_format['fs_post']) for signal_name, signal_format in self.signals_format.items()]
            record_duration = min(signal_durations)
            record_duration_clipped = record_duration - (record_duration % prediction_resolution)
            number_of_windows = max(1, int((record_duration_clipped - self.window) // (self.window * (1 - self.overlap)) + 1))

            # find signal info:
            if self.load_signal_in_RAM:
                for key, item in data.items():
                    data[key] = np.array(item).astype('float32')
                self.signals[record] = {"data": data}
            k = 1
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

            # Select events
            for label, event in enumerate(self.events_select_format):
                data = self.load_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                )
                self.events_select[record][event["name"]] = {
                    "data": np.array(data).astype('float32'),
                    "label": label,
                }

            for label, event in enumerate(events_format):

                events = self.load_events(
                    filename="{}/{}".format(h5_directory, record),
                    event=event,
                )

                """
                if use_mask:
                    for ed_key, ed_it in self.events_discard[record].items():
                        overlap = intersection_overlap(
                            np.array([ed_it['data'][0, :], ed_it['data'][0, :] + ed_it['data'][1, :]]).T,
                            np.array([events[0, :], events[0, :] + events[1, :]]).T)
                        if len(overlap) > 0:
                            max_iou = overlap.max(0)
                            keep_bool = (max_iou < (self.discard_threshold))
                            events = events[:, keep_bool]

                    for es_key, es_it in self.events_select[record].items():
                        overlap = intersection_overlap(
                            np.array([es_it['data'][0, :], es_it['data'][0, :] + es_it['data'][1, :]]).T,
                            np.array([events[0, :], events[0, :] + events[1, :]]).T)
                        if len(overlap) > 0:
                            max_iou = overlap.max(0)
                            keep_bool = (max_iou > (self.select_threshold))
                            events = events[:, keep_bool]
                """
                self.events[record][event["name"]] = {
                    "data": np.array(events).astype('float32'),
                    "label": label,
                }

                self.index_to_record_event[event["name"]].extend([{
                    "record": record,
                    "max_index": max(0, record_duration_clipped - self.window),
                    "data": np.array(events[:, n]).astype('float32')
                } for n in range(events.shape[1])])
        k = 1
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
                self.index_to_record = self.index_to_record[:index_val] #deepcopy(self.index_to_record[:index_val])
            elif dataset_type == 'train':
                self.index_to_record = self.index_to_record[index_val:] #deepcopy(self.index_to_record[index_val:])

    """
    def __len__(self):
        return len(self.index_to_record) // self.batch_size + 1

    def __getitem__(self, idx):
        idx = idx % int(len(self.index_to_record) // self.batch_size)
        signal_batch = {
            signal_name: np.zeros((self.batch_size,
                                   int(self.window * signal_format['fs_post']),
                                   signal_format['num_features'])).astype('float32')
            for signal_name, signal_format in self.signals_format[next(iter(self.signals_format.keys()))].items()}
        events_batch = []
        for counter, idx_ in enumerate(range(idx * self.batch_size, (idx + 1) * self.batch_size)):
            idx_ = idx_ % len(self.index_to_record)
            for signal_name, signal_format in self.signals_format.items():
                signal = self.get_signals(
                    record=self.index_to_record[idx_]['record'],
                    signal_name=signal_name,
                    index=self.index_to_record[idx_]['index'])
                if signal_format['batch_normalization']:
                    signal = normalizers[signal_format['batch_normalization']['type']](signal, **signal_format['batch_normalization']['args'])
                for trans in signal_format['transformations'].keys():
                    signal = self.transformations[trans](signal)
                signal_batch[signal_name][counter, :] = signal
            events = self.get_events(
                record=self.index_to_record[idx_]['record'],
                index=self.index_to_record[idx_]['index'])
            events_batch += [events]

        model_input = []
        for models in self.model_format:
            model_input += [np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)]
        return model_input, np.array(events_batch).astype('float32')
    """

    def __getitem__(self, idx):
        idx = idx % int(len(self.index_to_record) // self.batch_size)
        index_to_record = [self.index_to_record[idx_] for idx_ in range(idx * self.batch_size, (idx + 1) * self.batch_size) if idx_ < len(self.index_to_record)]
        signals, events = self.get_item(index_to_record=index_to_record)
        model_input = self.signal_model_prep(signal_batch=signals)
        return model_input, events

    def get_item(self, index_to_record):
        signal_batch = self.prep_signal_batch(len(index_to_record))
        events_batch = []
        for counter, itr in enumerate(index_to_record):
            for signal_name, signal_format in self.signals_format.items():
                signal = self.get_signals(record=itr['record'],
                                          signal_name=signal_name,
                                          index=itr['index'])
                if signal_format['batch_normalization']:
                    signal = normalizers[signal_format['batch_normalization']['type']](signal, **signal_format['batch_normalization']['args'])
                if self.mode == 'train':
                    for trans, item in signal_format['transformations'].items():
                        signal = self.transformations[trans](signal, **item)
                signal_batch[signal_name][counter, :signal.shape[0], :] = signal
            events = self.get_events(record=itr['record'],
                                     index=itr['index'])
            events_batch += [events]
        return signal_batch, np.array(events_batch).astype('float32')

    def prep_signal_batch(self, batch_size):
        signal_batch = {signal_name: np.zeros((batch_size,
                                               int(self.window * signal_format['fs_post']),
                                               signal_format['num_features'])).astype('float32')
                        for signal_name, signal_format in self.signals_format.items()}
        return signal_batch

    def signal_model_prep(self, signal_batch):
        model_input = []
        if isinstance(self.model_format, dict):
            sig = np.concatenate([signal_batch[signal_name] for signal_name in self.model_format['signals']], axis=-1)
            check_inf_nan(sig)
            model_input += [sig]
        else:
            for models in self.model_format:
                model_input += [np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)]
        return model_input

    def get_signals(self, record, signal_name, index):
        # TODO - just modify this to zeropadding!
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
            mask_non_event = (np.sum(events, axis=-1) == 0)  # mask if no event
            events[mask_any, :] = -1
            events[mask_non_event, :] = -1

        return events

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
                k = 1
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

        signal_batch = {signal_name: np.zeros((self.batch_size,
                                               int(self.window * signal_format['fs_post']),
                                               signal_format['num_features']))
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
                signal_batch[signal_name][counter, :signal.shape[0], :] = signal
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
                        model_input += [np.concatenate([signal_batch[signal_name] for signal_name in models['signals']], axis=-1)]
                yield model_input, np.array(start_times)
                signal_batch = {signal_name: np.zeros((self.batch_size,
                                                       int(self.window * signal_format['fs_post']),
                                                       signal_format['num_features']))
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


    def get_hypnogram_semantic(self, record):

        #for signal_name, signal_format in self.signals_format.items():
        #    signal_size = (self.signals[record]['data'][signal_name]).shape[0]
        #    signal_size_sec = int(signal_size / signal_format['fs_post'])
        #    break

        rec_dur = self.get_rec_dur(record)

        WAKE = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'wake'}), 0, -1).tolist()
        LIGHT = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'light'}), 0, -1).tolist()
        DEEP = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'deep'}), 0, -1).tolist()
        REM = np.moveaxis(self.get_record_events_by_label(record, event_label={'h5_path': 'rem'}), 0, -1).tolist()

        WAKE  = [[w[0] / rec_dur, (w[0] + w[1]) / rec_dur] for w in WAKE]
        LIGHT = [[l[0] / rec_dur, (l[0] + l[1]) / rec_dur] for l in LIGHT]
        DEEP = [[d[0] / rec_dur, (d[0] + d[1]) / rec_dur] for d in DEEP]
        REM = [[r[0] / rec_dur, (r[0] + r[1]) / rec_dur] for r in REM]

        hypnogram = np.zeros((rec_dur, 4))
        try:
            hypnogram[:, 0] = semantic_formating(rec_dur, WAKE)
            hypnogram[:, 1] = semantic_formating(rec_dur, LIGHT)
            hypnogram[:, 2] = semantic_formating(rec_dur, DEEP)
            hypnogram[:, 3] = semantic_formating(rec_dur, REM)
        except:
            k = 1
        return hypnogram

    def get_data_by_event(self, event_label, reference_point='start'):
        indexes = []
        for num, itre in enumerate(self.index_to_record_event[event_label]):
            if reference_point == 'start':
                index = itre['data'][0] - (self.window / 2)
            else:
                index = (itre['data'][0] + itre['data'][1]) - (self.window / 2)

            if index > 0 and index < itre['max_index']:
                itre['index'] = index
                indexes += [itre]

        signals, events = self.get_item(index_to_record=indexes)
        return signals, events

    def get_data_by_window(self, record='', model=None):
        if record:
            idx = [itr for itr in self.index_to_record if itr['record'] == record]
        else:
            idx = self.index_to_record
        records = [i_['record'] for i_ in idx]
        idxs = [i_['index'] for i_ in idx]
        signals, events = self.get_item(index_to_record=idx)
        return signals, events, records, idxs

    def get_data_by_record(self, model=None):
        sig_rec = []
        tar_rec = []
        pre_rec = []
        rec_dur = []

        for record in self.records:
            record_dur = [itr['max_index'] + self.window for itr in self.index_to_record if itr['record'] == record][0]

            sig = {key: np.zeros((int(record_dur * item['fs_post']), item['num_features']))
                      for key, item in self.signals_format.items()}
            tar = np.zeros((record_dur // self.prediction_resolution, self.number_of_events))
            pre = np.zeros((record_dur // self.prediction_resolution, self.number_of_events))

            # get data
            signals, events, record_, start_idxs = self.get_data_by_window(record=record)
            if model is not None:
                predictions = self.predict_on_data(model=model, signals=signals)

            for idx, start_idx in enumerate(start_idxs):
                for key, item in self.signals_format.items():
                    sig[key][int(start_idx * item['fs_post']):
                             int((start_idx + self.window) * item['fs_post']), :] = signals[key][idx, :]
                tar[start_idx // self.prediction_resolution:
                    (start_idx + self.window) // self.prediction_resolution, :] = events[idx, :]
                if model is not None:
                    pre[start_idx // self.prediction_resolution:
                        (start_idx + self.window) // self.prediction_resolution, :] = predictions[idx, :]

            sig_rec += [sig]
            tar_rec += [tar]
            rec_dur += [record_dur]
            if model is not None:
                pre_rec += [pre]

        return sig_rec, tar_rec, pre_rec, self.records, rec_dur


    def predict_on_data(self, model, signals):
        signal_size = signals[next(iter(signals.keys()))].shape[0]
        num_of_batches = int(np.ceil(signal_size / self.batch_size))
        prediction = np.zeros((signal_size, self.predictions_per_window, self.number_of_classes))
        for idx in range(num_of_batches):
            batch_dict = {}
            idx_start = idx * self.batch_size
            idx_stop = min((idx + 1) * self.batch_size, signal_size)
            for key, item in signals.items():
                batch_dict[key] = item[idx_start:idx_stop, :]
            model_input = self.signal_model_prep(signal_batch=batch_dict)
            prediction[idx_start:idx_stop, :] = model.predict(model_input) # TODO - test if this is correct
        return prediction


    def hypnogram_plot_helper(self, targets):
        self.hypnogram_position = {
            'mask': 0,
            'deep': 1,
            'light': 2,
            'rem': 3,
            'wake': 4
        }
        target_loc = np.argmax(targets, axis=1)
        hyp_labels = np.array([self.event_labels[tar] for tar in target_loc])
        hyp_labels[targets[:, 0] == -1] = 'mask'
        hyp_plot_position = [self.hypnogram_position[hyp] for hyp in hyp_labels]
        return hyp_plot_position

    def plot(self, filename='', save_dir='', signals=None, targets=None, predictions=None, extra_events=None,
             figsize=(5.5, 0.75), disp_HRV_features=False, window_len=30, plot_hypnogram=True,
             labelx=-0.07, labely=0.5, fontsize=6, labelpad=10):

        create_directory_tree(save_dir)

        if window_len < 300:
            factor = 1
            xlabel = 'seconds'
            steps = 10
        elif window_len < 3600 * 2:
            factor = 60
            xlabel = 'minutes'
            steps = 10
        else:
            factor = 3600
            xlabel = 'hours'
            steps = 1

        plt.rcParams.update({'font.size': fontsize})


        for key, item in self.signals_format.items():

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.05, 0, 0.90, 1]) # [left, bottom, width, height]
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)

            t = np.arange(start=0, stop=(signals[key].shape[0]), step=1) / (item['fs_post'] * factor)
            # xaxis = np.arange(start=0, stop=(signals[key].shape[0]) / (item['fs_post']), step=(signals[key].shape[0]) / (item['fs_post']) // 10)
            xaxis = np.arange(start=0, stop=window_len // factor + 1, step=steps)
            k = 1
            if key[-4:] == 'spec' or key == 'feature_based_acc':

                # extract spectrogram
                S = np.swapaxes(signals[key], axis1=1, axis2=0)
                S = np.flipud(S)
                S = (S - S.min()) / (S.max() - S.min())
                # TODO - du normalizere her!

                if key[-12:] == 'cwt_new_spec':

                    wavelet = 'morl'
                    scales = np.logspace(start=1, stop=7, num=64, base=2)
                    f = pywt.scale2frequency(wavelet, scales) * 32

                    im = ax.imshow(S, extent=(np.amin(t), np.amax(t), np.amin(f), np.amax(f)),
                                    aspect='auto',
                                    vmin=np.quantile(S, q=0.05), vmax=np.quantile(S, q=0.99))
                    #np.logspace(start=-1, stop=np.log10(f_max), num=8, base=10)                                # vmin=-10, vmax=10)

                    # y axis
                    yticks = np.arange(start=f[0], stop=f[-1], step=(f[-1] - f[0]) / 8)
                    yaxis = f[::f.shape[0] // 8]
                    ax.yaxis.set_ticks(yticks)
                    ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                    ax.yaxis.set_ticklabels(['{:.1f}'.format(y) for y in yaxis], fontsize=fontsize)
                    ax.set_ylabel(r'freq (Hz)', size=fontsize, rotation=90, labelpad=labelpad)
                    ax.yaxis.set_label_coords(labelx, labely)

                else:
                    try:
                        f_min = item['preprocessing'][-1]['args']['f_min'] #+ 1
                        f_max = item['preprocessing'][-1]['args']['f_max'] #+ 1
                    except:
                        f_min = -3
                        f_max = 3
                        #S = np.swapaxes(S, axis1=1, axis2=0)
                        #S = np.flipud(S)
                    linear = True
                    if linear:
                        yaxis = list(np.arange(start=f_min, stop=f_max, step=(f_max + 1 - f_min) // 2))
                    else:
                        yaxis = list(np.logspace(start=-3, stop=np.log2(f_max), num=8, base=2))
                        ax.set_yscale("log", basey=2) #, subsy=list(range(6))) #)

                    im = ax.imshow(S, extent=(np.amin(t), np.amax(t), f_min, f_max), aspect='auto',
                                   vmin=np.quantile(S, q=0.05), vmax=np.quantile(S, q=0.99))

                    # yaxis
                    ax.yaxis.set_ticks(yaxis)
                    ax.yaxis.set_tick_params(which='major', size=1, width=0.5, direction='out')
                    ax.yaxis.set_ticklabels(['{:.1f}'.format(y) for y in yaxis], fontsize=fontsize)
                    ax.set_ylabel(r'freq (Hz)', size=fontsize, rotation=90, labelpad=labelpad)
                    ax.yaxis.set_label_coords(labelx, labely)

                # x axis
                ax.xaxis.set_ticks(xaxis)
                ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
                ax.set_xlabel('time ({})'.format(xlabel), fontsize=fontsize)

                # colorbar
                cax = fig.add_axes([0.96, 0, 0.02, 1])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=fontsize)
                cax.set_ylabel('spectral intensity (a.u.)', fontsize=fontsize, labelpad=labelpad)

            else:
                # int_min = np.floor(signals[key].min())
                # int_max = np.ceil(signals[key].max())
                x_max = (targets.shape[0]) * self.prediction_resolution

                #sig_norm = (signals[key] - signals[key].min()) / (signals[key].max() - signals[key].min() + sys.float_info.epsilon)
                sig_norm = signals[key]
                ax.plot(t, sig_norm, label=key, linewidth=.4)

                #xaxis = np.arange(start=0, stop=t[-1], step=60 * 10)
                #xaxis = np.append(t[::t.shape[0] // 8], t[-1])
                ax.xaxis.set_ticks(xaxis)
                ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
                plt.xlabel('time ({})'.format(xlabel), fontsize=fontsize)
                #plt.xlim([xaxis[0], xaxis[-1]])
                # plt.xlim([0, t[-1]])
                plt.xlim([0, window_len // factor + 1])

                # y axis
                #yaxis = [0, 0.25, 0.5, 0.75, 1.00]

                ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                #ax.yaxis.set_ticks(yaxis)
                #ax.yaxis.set_ticklabels(['{:.1f}'.format(y) for y in yaxis], fontsize=fontsize)
                ax.set_ylabel('signal intensity (a.u)', size=fontsize, rotation=90, labelpad=labelpad)
                #ax.set_ylim([-0.05, 1.05])
                ax.yaxis.set_label_coords(labelx, labely)

                ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)

                if key == 'PPG_features' or  key == 'PPG_features_downsampled':
                    plt.legend(('FM', 'AM', 'BW'), loc=2, borderaxespad=0., frameon=True, fontsize=fontsize - 1)
                if key == 'ACC' or key == 'ACC_raw':
                    plt.legend(('X', 'Y', 'Z'), loc=2, borderaxespad=0., frameon=True, fontsize=fontsize - 1)

            # save figure
            num = 0
            while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, key, num)): num += 1
            plt.savefig('{}\\{}{}_{}.png'.format(save_dir, filename, key, num), transparent=True, dpi=600, bbox_inches='tight')
            #plt.savefig('{}\\{}{}_{}.pdf'.format(save_dir, filename, key, num), transparent=True, dpi=600, bbox_inches='tight')

        if predictions is not None:

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.05, 0, 0.90, 1]) # [left, bottom, width, height]
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ts_event = np.arange(start=0, stop=(predictions.shape[0]) * self.prediction_resolution,
                                 step=self.prediction_resolution) / factor
            if plot_hypnogram:
                hypnogram = self.hypnogram_plot_helper(predictions)
                ax.plot(ts_event, hypnogram, label='prediction', linewidth=0.5)
            else:
                ax.plot(ts_event, predictions, linewidth=0.5)

            # xaxis
            ax.xaxis.set_ticks(xaxis)
            ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
            ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
            ax.set_xlabel('time ({})'.format(xlabel), fontsize=fontsize)
            # ax.set_xlim([0, t[-1]])
            plt.xlim([0, window_len // factor + 1])

            # y axis
            # ax.yaxis.set_ticks(yaxis)
            if plot_hypnogram:
                ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                ax.yaxis.set_ticks(list(self.hypnogram_position.values()))
                ax.yaxis.set_ticklabels(list(self.hypnogram_position.keys()), fontsize=fontsize)
                #ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
                ax.set_ylabel(r'hypnogram', size=fontsize, rotation=90, labelpad=labelpad)
                ax.set_ylim([-0.05, 4.05])
                ax.yaxis.set_label_coords(labelx, labely)
            else:
                ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                ax.yaxis.set_ticks(list([0, 1]))
                ax.yaxis.set_ticklabels(['no event', 'event'], fontsize=fontsize)
                ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
                ax.set_ylabel(r'event', size=fontsize, rotation=90, labelpad=labelpad)
                ax.set_ylim([-0.05, 1.05])
                ax.yaxis.set_label_coords(labelx, labely)

            num = 0
            while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'prediction', num)): num += 1
            # save predictions
            plt.savefig('{}\\{}{}_{}.png'.format(save_dir, filename, 'prediction', num), transparent=True, dpi=600,
                        bbox_inches='tight')
            #plt.savefig('{}\\{}{}_{}.pdf'.format(save_dir, filename, 'prediction', num), transparent=True, dpi=600,
            #            bbox_inches='tight')

        if targets is not None:

            # while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num)): num += 1
            # if not os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num)):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.05, 0, 0.90, 1]) # [left, bottom, width, height]
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ts_event = np.arange(start=0, stop=(targets.shape[0]) * self.prediction_resolution,
                                 step=self.prediction_resolution) / factor
            if plot_hypnogram:
                hypnogram = self.hypnogram_plot_helper(targets)
                ax.plot(ts_event, hypnogram, label='target', linewidth=0.5)
            else:
                ax.plot(ts_event, targets, linewidth=0.5)

            # x axis
            ax.xaxis.set_ticks(xaxis)
            ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
            ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
            ax.set_xlabel('time ({})'.format(xlabel), fontsize=fontsize)
            #ax.set_xlim([0, t[-1]])
            plt.xlim([0, window_len // factor + 1])

            # y axis
            # ax.yaxis.set_ticks(yaxis)
            if plot_hypnogram:
                ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                ax.yaxis.set_ticks(list(self.hypnogram_position.values()))
                ax.yaxis.set_ticklabels(list(self.hypnogram_position.keys()), fontsize=fontsize)
                #ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
                ax.set_ylabel(r'hypnogram', size=fontsize, rotation=90, labelpad=labelpad)
                ax.set_ylim([-0.05, 4.05])
                ax.yaxis.set_label_coords(labelx, labely)
            else:
                ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
                ax.yaxis.set_ticks(list([0, 1]))
                ax.yaxis.set_ticklabels(['no event', 'event'], fontsize=fontsize)
                ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
                ax.set_ylabel(r'events', size=fontsize, rotation=90, labelpad=labelpad)
                ax.set_ylim([-0.05, 1.05])
                ax.yaxis.set_label_coords(labelx, labely)

            num = 0
            while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num)): num += 1
            # save target
            plt.savefig('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num), transparent=True, dpi=600,
                        bbox_inches='tight')
            #plt.savefig('{}\\{}{}_{}.pdf'.format(save_dir, filename, 'target', num), transparent=True, dpi=600,
            #            bbox_inches='tight')

    def plot_hypnogram(self, hypnogram, filename = '', save_dir = '', window_len=None, figsize=(5.5, 0.75)):

        create_directory_tree(save_dir)

        fontsize = 6
        labelpad = 0
        num = 0
        labelx = -0.07
        labely = 0.5
        labelpad = 0

        if window_len < 300:
            factor = 1
            xlabel = 'seconds'
            steps = 10
        elif window_len < 3600 * 2:
            factor = 60
            xlabel = 'minutes'
            steps = 10
        else:
            factor = 3600
            xlabel = 'hours'
            steps = 1

        plt.rcParams.update({'font.size': fontsize})

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0, 0.90, 1])  # [left, bottom, width, height]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # x axis
        # xaxis = np.arange(start=0, stop=window_len // factor, step=steps)
        xaxis = np.arange(start=0, stop=window_len // factor + 1, step=steps)

        # while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num)): num += 1
        # if not os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'target', num)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0, 0.90, 1]) # [left, bottom, width, height]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ts_event = np.arange(start=0, stop=(hypnogram.shape[0]) * self.prediction_resolution,
                             step=self.prediction_resolution) / factor

        hypnogram = self.hypnogram_plot_helper(hypnogram)
        ax.plot(ts_event, hypnogram, label='target', linewidth=0.5)

        # x axis
        ax.xaxis.set_ticks(xaxis)
        ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
        ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
        ax.set_xlabel('time ({})'.format(xlabel), fontsize=fontsize)
        # ax.set_xlim([0, t[-1]])
        plt.xlim([0, window_len // factor + 1])

        ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
        ax.yaxis.set_ticks(list(self.hypnogram_position.values()))
        ax.yaxis.set_ticklabels(list(self.hypnogram_position.keys()), fontsize=fontsize)
        #ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
        ax.set_ylabel(r'hypnogram', size=fontsize, rotation=90, labelpad=labelpad)
        ax.set_ylim([-0.05, 4.05])
        ax.yaxis.set_label_coords(labelx, labely)

        # save stuff
        plt.savefig('{}\\{}{}_{}.png'.format(save_dir, filename, 'hypnogram', num), transparent=True, dpi=300,
                    bbox_inches='tight')
        print('hypnogram ')
        #plt.savefig('{}\\{}{}_{}.pdf'.format(save_dir, filename, 'hypnogram', num), transparent=True, dpi=300,
        #            bbox_inches='tight')


    def plot_events(self, events, labels, filename = '', save_dir = '', window_len=None, figsize=(5.5, 0.75)):

        create_directory_tree(save_dir)

        fontsize = 6
        labelpad = 0
        num = 0
        labelx = -0.07
        labely = 0.5
        labelpad = 0

        num_format = len(events)

        if window_len < 300:
            factor = 1
            xlabel = 'seconds'
            steps = 10
        elif window_len < 3600 * 2:
            factor = 60
            xlabel = 'minutes'
            steps = 10
        else:
            factor = 3600
            xlabel = 'hours'
            steps = 1

        plt.rcParams.update({'font.size': fontsize})

        # while os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'extra', num)): num += 1
        if not os.path.isfile('{}\\{}{}_{}.png'.format(save_dir, filename, 'extra', num)):

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.05, 0, 0.90, 1]) # [left, bottom, width, height]
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)

            for count, event in enumerate(events):
                for n in event:
                    ax.axvspan(n[0] / factor, n[1] / factor , count / num_format + 0.05, (count + 1) / num_format - 0.05, alpha=1)
                    # ax.axvspan(n[0] / factor, n[1] / factor, count + 0.05, (count + 1) - 0.05, alpha=1)

            # x axis
            # xaxis = np.arange(start=0, stop=window_len // factor, step=steps)
            xaxis = np.arange(start=0, stop=window_len // factor + 1, step=steps)
            # xaxis = np.append(t[::t.shape[0] // 8], t[-1])
            ax.xaxis.set_ticks(xaxis)
            ax.xaxis.set_ticklabels([int(x) for x in xaxis], fontsize=fontsize)
            #ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
            plt.xlabel('time ({})'.format(xlabel), fontsize=fontsize)
            plt.xlim([0, window_len // factor + 1])

            ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)

            # y axis
            # ax.yaxis.set_ticks(yaxis)
            ax.yaxis.set_tick_params(which='major', size=2, width=0.5, direction='out')
            ax.yaxis.set_ticks([x + 0.5 for x in range(num_format)])
            ax.yaxis.set_ticklabels(labels + ['']*(num_format - len(labels)), fontsize=fontsize)
            # ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
            ax.set_ylabel(r'event', size=fontsize, rotation=90, labelpad=labelpad)
            ax.set_ylim([0 - 0.05, num_format + 0.05])
            ax.yaxis.set_label_coords(labelx, labely)



            # save stuff
            plt.savefig('{}\\{}{}_{}.png'.format(save_dir, filename, 'extra', num), transparent=True, dpi=300,
                        bbox_inches='tight')
            #plt.savefig('{}\\{}{}_{}.pdf'.format(save_dir, filename, 'extra', num), transparent=True, dpi=300,
            #            bbox_inches='tight')


    def plot_spectrogram_and_label(self, S, e, f_min, f_max, t_min, t_max, text=''):

        gs = gridspec.GridSpec(2, 1)
        fig = plt.figure(figsize=(30, 8), dpi=100)
        fig.suptitle('spectrogram', fontsize=14)

        ax = fig.add_subplot(gs[0])

        S = np.swapaxes(S, axis1=1, axis2=0)
        S = np.flipud(S)
        ax.imshow(S, extent=(t_min, t_max, f_min, f_max), cmap=cm.hot, aspect='auto')
        ax.set_ylabel(r'freq (Hz)', size=12)
        ax.set_xlabel(r'time (s)', size=12)

        ax = fig.add_subplot(gs[1], sharex=ax)
        t = np.arange(t_min, t_max)
        ax.plot(t, e)
        ax.set_ylim([-1.05, 1.05])
        ax.set_xlim([t_min, t_max])

        plt.savefig('E:\\Mic_study\\figures\\' + text + '.png')

    def compile_event(self, event_label='', reference_point='start', model=None, plot_individual=False, name=''):

        signals_compiled = []
        events_compiled = []
        predictions = []
        prediction = None

        for event_num, event_inst in enumerate(self.index_to_record_event[event_label]):
            if reference_point == 'start':
                index = event_inst['data'][0] - (self.window / 2)
            else:
                index = (event_inst['data'][0] + event_inst['data'][1]) - (self.window / 2)

            if index > 0 and index < event_inst['max_index']:
                for signal_name, signal_format in self.signals_format.items():
                    signal = self.get_signals(record=event_inst['record'],
                                              signal_name=signal_name,
                                              index=event_inst['index'])
                    if signal_format['batch_normalization']:
                        signal = normalizers[self.batch_normalization['type']](signal,
                                                                               **self.batch_normalization['args'])
                event = self.get_events(record=event_inst['record'],
                                        index=event_inst['index'])
                if model is not None:
                    prediction = model.predict(np.array([signal]))[0, :, :]
                    predictions += [prediction]
                if plot_individual:
                    if len(self.input_shape) == 2:
                        self.plot(name=str(event_num) + '_' + name,
                                  path=event_inst['record'] + '_' + event_label,
                                  signals=signal,
                                  targets=event,
                                  predictions=prediction)
                    else:
                        self.plot(str(event_num) + '_' + name,
                                  path=event_inst['record'] + '_' + event_label,
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
            for signal_name, signal_format in self.signals_format.items():
                signal = self.get_signals(record=event_inst['record'],
                                          signal_name=signal_name,
                                          index=event_inst['index'])
                if signal_format['batch_normalization']:
                    signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
            event = self.get_events(record=event_inst['record'],
                                    index=event_inst['index'])
            if model is not None:
                prediction = model.predict(np.array([signal]))[0, :, :]
                predictions += [prediction]
            if plot_individual:
                if len(self.input_shape) == 2:
                    self.plot(name=name + ' ' + str(event_num),
                              path=event_inst['record'],
                              signals=signal,
                              targets=event,
                              predictions=prediction)
                else:
                    self.plot(name=str(event_num),
                              path=event_inst['record'],
                              spectrograms=signal,
                              targets=event,
                              predictions=prediction)

            signals_compiled += [signal]
            events_compiled += [event]

        return np.array(signals_compiled), np.array(events_compiled), np.array(predictions)

    def compile_record(self, record, event_label='', reference_point='start', model=None, plot_individual=False, name=''):
        # TODO - Not done yet.
        assert self.overlap == 0

        signals_compiled = []
        events_compiled = []
        mask_compiled = []
        predictions = []
        prediction = None

        for event_num, event_inst in enumerate(self.index_to_record):
            if event_inst['record'] == record:
                for signal_name, signal_format in self.signals_format.items():
                    signal = self.get_signals(record=event_inst['record'],
                                              signal_name=signal_name,
                                              index=event_inst['index'])
                    if signal_format['batch_normalization']:
                        signal = normalizers[self.batch_normalization['type']](signal, **self.batch_normalization['args'])
                event = self.get_events(record=event_inst['record'],
                                        index=event_inst['index'])

                if model is not None:
                    prediction = model.predict(np.array([signal]))[0, :, :]
                    predictions += [prediction]
                if plot_individual:
                    if len(self.input_shape) == 2:
                        self.plot(name=name + ' ' + str(event_num),
                                  path=event_inst['record'],
                                  signals=signal,
                                  targets=event,
                                  predictions=prediction)
                    else:
                        self.plot(name=str(event_num),
                                  path=event_inst['record'],
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
        hypnogram = self.get_hypnogram_semantic(record=record)


        return signals, events, predictions, hypnogram
