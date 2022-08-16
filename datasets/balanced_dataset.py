import numpy as np
import random
from matplotlib import gridspec
from time import time
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

from utils import semantic_formating, any_formating
from datasets import Dataset
from signal_processing import regularizers, normalizers
from utils import get_h5_data, get_h5_events, check_inf_nan, get_h5_auxiliary


class BalancedDataset(Dataset):
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
                 events_discard_format=[],
                 events_select_format=[],
                 events_format=None,
                 prediction_resolution=1,
                 overlap=0.5,
                 h5_directory_auxiliary='',
                 minimum_overlap=0.5,
                 batch_size=64,
                 mode='inference',
                 transformations_factor=0.1,
                 val_rate=0,
                 dataset_type='',
                 discard_threshold=10,
                 select_threshold=10,
                 use_mask=True,
                 cache_data=True,
                 load_signal_in_RAM=True,
                 batch_len_multiplier=1):

        super().__init__(records=records,
                         h5_directory=h5_directory,
                         signals_format=signals_format,
                         h5_directory_auxiliary=h5_directory_auxiliary,
                         window=window,
                         model_format=model_format,
                         events_discard_format=events_discard_format,
                         events_select_format=events_select_format,
                         number_of_channels=number_of_channels,
                         events_format=events_format,
                         overlap=overlap,
                         prediction_resolution=prediction_resolution,
                         minimum_overlap=minimum_overlap,
                         batch_size=batch_size,
                         mode=mode,
                         transformations_factor=transformations_factor,
                         discard_threshold=discard_threshold,
                         select_threshold=select_threshold,
                         val_rate=val_rate,
                         dataset_type=dataset_type,
                         use_mask=use_mask,
                         cache_data=cache_data,
                         load_signal_in_RAM=load_signal_in_RAM,
                         batch_len_multiplier=batch_len_multiplier)

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

        event = np.random.choice((self.number_of_classes + 1),
                                 p=self.event_probabilities + [1 - sum(self.event_probabilities)])

        if event == (self.number_of_classes):
            # get random segment from file
            random_event = random.sample(self.index_to_record, 1)[0]
            index = np.random.randint(random_event['max_index'])
            signal_data, events_data = self.get_sample(random_event['record'], int(index))
            while np.sum(events_data) != 0:
                random_event = random.sample(self.index_to_record, 1)[0]
                index = np.random.randint(random_event['max_index'])
                signal_data, events_data = self.get_sample(random_event['record'], int(index))
        else:
            # get random event from random file.
            random_event = random.sample(self.index_to_record_event[self.events_format[event]['name']], 1)[0]
            try:
                if (self.window <= random_event["data"][1]) or random_event["data"][0] == 0:
                    index = random_event["data"][0]
                elif random_event["data"][0] >= random_event['max_index']:
                    index = random_event['max_index']
                else:

                    index = np.random.randint(
                        low=max(0, random_event["data"][0] - (self.window - random_event["data"][1])),
                        high=min(random_event['max_index'], random_event["data"][0]))
            except:
                print(random_event)
            signal_data, events_data = self.get_sample(random_event['record'], int(index))

        return signal_data, events_data

class CombineDatasets(Dataset):
    """Extract data and events from h5 files and provide efficient way to retrieve windows with
    their corresponding events.

    args
    ====

    h5_directory:
        Location of the generic h5 files.
    signals:
        The signals from the h5 we want to include together with their normalization
    """

    def __init__(self,
                 datasets,
                 window,
                 number_of_channels,
                 model_format=None,
                 events_format=None,
                 overlap=0.5,
                 prediction_resolution=1,
                 minimum_overlap=0.5,
                 batch_size=64,
                 h5_directory_auxiliary='',
                 mode='inference',
                 transformations_factor=0.1,
                 discard_threshold=10,
                 events_discard_format=None,
                 events_select_format=None,
                 val_rate=0,
                 use_mask=True,
                 dataset_type='',
                 cache_data=False,
                 load_signal_in_RAM=True,
                 batch_len_multiplier=1):

        super().__init__(records=[],
                         h5_directory=None,
                         signals_format={},
                         events_discard_format=events_discard_format,
                         events_select_format=events_select_format,
                         h5_directory_auxiliary=h5_directory_auxiliary,
                         window=window,
                         model_format=model_format,
                         number_of_channels=number_of_channels,
                         events_format=events_format,
                         overlap=overlap,
                         prediction_resolution=prediction_resolution,
                         minimum_overlap=minimum_overlap,
                         batch_size=batch_size,
                         mode=mode,
                         transformations_factor=transformations_factor,
                         discard_threshold=discard_threshold,
                         val_rate=val_rate,
                         use_mask=use_mask,
                         dataset_type=dataset_type,
                         cache_data=cache_data,
                         load_signal_in_RAM=load_signal_in_RAM,
                         batch_len_multiplier=batch_len_multiplier)

        self.mode = mode
        self.number_of_databases = len(datasets)
        self.dataset_names = [database['name'] for database in datasets]
        self.datasets_probabilities = [database['probability'] for database in datasets]
        self.batch_len_multiplier = batch_len_multiplier

        self.signals = {}
        self.events = {}
        self.events_discard = {}
        self.index_to_record = {}
        self.index_to_record_event = {}
        self.number_of_samples = 0
        self.events_format = events_format

        # avoiding having signals in RAM
        self.h5_directories = {}
        self.get_db_data = {}
        self.signals_format = {}

        for dataset in datasets:
            self.signals[dataset['name']] = dataset['dataset'].signals
            self.events[dataset['name']] = dataset['dataset'].events
            self.index_to_record[dataset['name']] = dataset['dataset'].index_to_record
            self.index_to_record_event[dataset['name']] = dataset['dataset'].index_to_record_event
            self.events_discard[dataset['name']] = dataset['dataset'].events_discard
            self.h5_directories[dataset['name']] = dataset['dataset'].h5_directory
            self.get_db_data[dataset['name']] = dataset['dataset'].load_data
            self.signals_format[dataset['name']] = dataset['dataset'].signals_format

    def __len__(self):
        iterations = []
        for dataset_num in range(self.number_of_databases):
            iterations += [len(self.index_to_record[self.dataset_names[dataset_num]])]
        max_len = 10000
        return min((int(sum(iterations) * self.batch_len_multiplier) // (self.batch_size)), max_len)

    def __getitem__(self, idx):

        dataset_names, records, indexes = zip(*[self.find_balanced_index() for counter in range(self.batch_size)])
        signals, events = self.get_item_datasets(dataset_names, records, indexes)
        model_input = self.signal_model_prep(signal_batch=signals)

        if len(self.h5_directory_auxiliary) > 0:
            auxiliary = self.get_auxiliary_item_datsets(dataset_names, records, indexes)
            return [model_input, auxiliary], events

        return model_input, events

    def final_prep(self, dataset_names, records, indexes):
        signals, events = self.get_item_datasets(dataset_names, records, indexes)
        model_input = self.signal_model_prep(signal_batch=signals)

        if len(self.h5_directory_auxiliary) > 0:
            auxiliary = self.get_auxiliary_item_datsets(dataset_names, records, indexes)
            return [model_input, auxiliary], events
        return model_input, events


    def get_item_datasets(self, dataset_names, records, indexes):
        signal_batch = {signal_name: np.zeros([self.batch_size,
                                               int(self.window * signal_format['fs_post'])] + signal_format[
                                                  'dimensions'])
                        for signal_name, signal_format in self.signals_format[next(iter(self.signals_format.keys()))].items()}
        events_batch = []

        for counter, (dataset_name, record, index) in enumerate(zip(dataset_names, records, indexes)):
            for signal_name, signal_format in self.signals_format[next(iter(self.signals_format.keys()))].items():
                signal = self.get_signal_database(dataset_name, record, signal_name, index)
                if signal_format['batch_normalization']:
                    signal = normalizers[signal_format['batch_normalization']['type']](signal, **
                    signal_format['batch_normalization']['args'])
                if self.mode == 'train':
                    for trans, item in signal_format['transformations'].items():
                        signal = self.transformations[trans](signal, **item)
                signal_batch[signal_name][counter, :signal.shape[0], :] = np.reshape(signal,
                                                                                 [signal.shape[0]] + signal_format[
                                                                                     'dimensions'])

            events_batch += [self.get_events_database(dataset_name, record, index)]

        """
        model_input = []
        if isinstance(self.model_format, dict):
            sig = np.concatenate([signal_batch[signal_name] for signal_name in self.model_format['signals']], axis=-1)
            check_inf_nan(sig)
            model_input += [sig]
        else:
            for models_old in self.model_format:
                sig = np.concatenate([signal_batch[signal_name] for signal_name in models_old['signals']], axis=-1)
                check_inf_nan(sig)
                model_input += [sig]
        k = 1
        """
        return signal_batch, np.array(events_batch).astype('float32')

    def get_auxiliary_item_datsets(self, dataset_names, records, indexes):
        auxiliary_batch = []
        for counter, (dataset_name, record, index) in enumerate(zip(dataset_names, records, indexes)):
            auxiliary_batch += [self.get_auxiliary_database(database_name=dataset_name,
                                                            record=record,
                                                            index=index)]
        return np.array(auxiliary_batch).astype('float32')


    def get_signal_database(self, database_name, record, signal_name, index):
        fs = self.signals_format[database_name][signal_name]['fs_post']
        if self.load_signal_in_RAM:
            signal_data = self.signals[database_name][record]["data"][signal_name][int(index * fs): int((index + self.window) * fs), :]
        else:
            data = self.get_db_data[database_name](filename="{}/{}".format(self.h5_directories[database_name], record), signals=self.signals_format[database_name])
            signal_data = data[signal_name][int(index * fs): int((index + self.window) * fs), :]
        return signal_data

    def get_events_database(self, database_name, record, index):
        events = self.get_events_sample(index=index, events=self.events[database_name][record], minimum_overlap=self.minimum_overlap)
        if self.use_mask:
            masks = self.get_events_sample(index=index, events=self.events_discard[database_name][record], minimum_overlap=0, any_overlap=True)
            mask_any = (np.sum(masks, axis=-1) > 0)
            #mask_non_event = (np.sum(events, axis=-1) == 0) # mask if no event
            events[mask_any, :] = -1
            #events[mask_non_event, :] = -1
        return events

    def get_auxiliary_database(self, database_name, record, index):
        data, fs = get_h5_auxiliary(filename="{}\\{}\\{}".format(self.h5_directories[database_name], self.h5_directory_auxiliary, record),
                                    labels=['wake', 'light', 'deep', 'rem'])
        x_aux = data[int(index * fs): int((index + self.window) * fs), :]
        if x_aux.shape[0] != int(self.window * fs):
            print(database_name)
            print(record)
            print(index)
            print(x_aux.shape[0])
            print(int(self.window * fs))
        assert(x_aux.shape[0] == int(self.window * fs))

        return x_aux

    def find_balanced_index(self):
        dataset = np.random.choice((self.number_of_databases), p=self.datasets_probabilities)
        database_name = self.dataset_names[dataset]
        event = np.random.choice((self.number_of_classes + 1), p=self.event_probabilities + [1 - sum(self.event_probabilities)])

        if event == (self.number_of_classes): # get random segment from file
            random_event = random.sample(self.index_to_record[database_name], 1)[0]
            if random_event['max_index'] > 0:
                index = np.random.randint(random_event['max_index'])
            else:
                index = 0
            #events_data = self.get_events_database(database_name, random_event['record'], int(index))
            #while np.sum(events_data) != 0:
            #    random_event = random.sample(self.index_to_record[database_name], 1)[0]
            #    index = np.random.randint(random_event['max_index'])
            #    events_data = self.get_events_database(database_name, random_event['record'], int(index))
        else: # get random event from random file.
            random_event = random.sample(self.index_to_record_event[database_name][self.events_format[event]['name']], 1)[0]
            try:
                if (self.window <= random_event["data"][1]) or random_event["data"][0] == 0:
                    index = random_event["data"][0]
                elif random_event["data"][0] >= random_event['max_index']:
                    index = random_event['max_index']
                else:
                    index = np.random.randint(
                        low=max(0, random_event["data"][0] - (self.window - random_event["data"][1])),
                        high=min(random_event['max_index'], random_event["data"][0]))
            except:
                print(random_event)
        # todo - index should be rounded down to nearest prediction resolutoin size:
        index = index - index % self.prediction_resolution
        return database_name, random_event['record'], int(index)
