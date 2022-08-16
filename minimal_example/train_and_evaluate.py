# Import section
# ======================================================================================================================
from datasets import BalancedDataset, get_train_validation_test



# Parameters section
# ======================================================================================================================
data_directory = '../data/h5'


# dataset creation
# ======================================================================================================================
import tensorflow
import numpy as np



seed = 2022

train, validation, test = get_train_validation_test(data_directory,
                                                    percent_test=40,
                                                    percent_validation=20,
                                                    seed=seed)

print('training set: {}'.format(len(train)))
print('eval set: {}'.format(len(validation)))
print('test set: {}'.format(len(test)))


# Pre-processing
# ======================================================================================================================
# A modality can have an arbitrary number of preprocessing steps.

signals_format = {
    'ACC_merge': {
        'h5_path': 'acc_signal',
        'channel_idx': [0, 1, 2],
        'preprocessing': [
            {
                'type': 'median',
                'args': {
                    'window_size': 30
                }
            },
            {
                'type': 'iqr_normalization_adaptive',
                'args': {
                    'median_window': 300,
                    'iqr_window': 300
                }
            },
            {
                'type': 'clip_by_iqr',
                'args': {
                    'threshold': 20
                }
            },
            {
                'type': 'cal_psd',
                'args': {
                    'window': 10 * 32,  # 1/21, 1/35
                    'noverlap': 8 * 32,  # 1/42, 1/75
                    'nfft': int(2 ** np.ceil(np.log2(10 * 32))),
                    'f_min': 0,
                    'f_max': 6,
                    'f_sub': 3
                }
            }

        ],
        'batch_normalization': {},
        'transformations': {
            #'shuffle_features': {},
            #'replace_with_noise': {},
            #'time_mask_USleep': {}
        },
        'add': True,
        'fs_post': 32,
        'dimensions': [int(2 ** np.ceil(np.log2(10 * 32)) / 32 * (6 - 0)) // 3, 1]
    },
    'ppg_signal': {
        'h5_path': 'ppg_signal',
        'channel_idx': [0],
        'preprocessing': [
            {
                'type': 'zscore',
                'args': {}
            },
            {
                'type': 'change_PPG_direction',
                'args': {}
            },
            {
                'type': 'iqr_normalization_adaptive',
                'args': {
                    'median_window': 301,
                    'iqr_window': 301
                }
            },
            {
                'type': 'clip_by_iqr',
                'args': {
                    'threshold': 20
                }
            },
            {
                'type': 'cal_psd',
                'args': {
                    'window': 10 * 32,  # 1/21, 1/35
                    'noverlap': 8 * 32,  # 1/42, 1/75
                    'nfft': int(2 ** np.ceil(np.log2(10 * 32))),
                    'f_min': 0.1,
                    'f_max': 2.1,
                    'f_sub': 1
                }
            }
        ],
        'batch_normalization': {},
        'transformations': {
            #'image_translation': {},
            #'time_mask': {},
            #'freq_mask': {},
        },
        'add': False,
        'fs_post': 1,
        'dimensions': [int(2 ** np.ceil(np.log2(10 * 32)) / 32 * (2.1 - 0.1)) // 1, 1]
    }
}


dataset_parameters = {
    "window": 30 * 2 ** 10,
    "prediction_resolution": 30,
    "overlap": 0.25,
    "minimum_overlap": 0.1,
    "batch_size": 2,
    "use_mask": True,
    "load_signal_in_RAM": True
}

dataset = BalancedDataset(**dataset_parameters)



# Load model
# ======================================================================================================================
#
model_params = {
    'init_filter_num': 16,
    'filter_increment_factor': 2 ** (1 / 3),
    'max_pool_size': (2, 2),
    'depth': 10,
    'kernel_size': (16, 3)
}


# Train model
# ======================================================================================================================

train_params = {
    'epochs': 100,
    'initial_epoch': 0,
    'patience': 18,
    'reduce_patience': 6,
    'monitor': 'cohens_kappa',
    'loss': {
        'type': 'weighted_loss',
        'args': {'alpha': 0.35, 'gamma': 2}
    },
    'learning_rate': 1e-4,
    'load_model_name': ''
}

# TODO - show in-line training (verbose).


# Evaluate model.
# ======================================================================================================================
# TODO - predict on test set
# TODO - visualize on test set (both target and predicted)
# TODO - calculate score.


# TODO - consider remove transposedConv, such that we upsample instead - look transConv i fixed to one size!

def train_execution(experiment, idx=0, load_model_name=''):

    experiment.create_directory_trees()
    train_parameters = experiment.get_train_params() # TODO - change this to get_params(group)
    train_parameters['model_name'] += '_{}'.format(str(idx))
    model = experiment.get_model(load_model_name=load_model_name)
    train_parameters.update({'model': model})
    #model.summary()

    for dataset in experiment._datasets.keys():
        print(' ')
        print(dataset)
        unique_train_recs = set([idx['record'] for idx in train_parameters['train_dataset'].index_to_record[dataset]])
        print('train_records', unique_train_recs)
        unique_validation_recs = set([idx['record'] for idx in train_parameters['validate_dataset'].index_to_record[dataset]])
        print('validation_records', unique_validation_recs)

    history = train(**train_parameters)
    idx = np.argmax(history.history['val_' + train_parameters['monitor']])
    best_model_name = train_parameters['model_name'] + '.{:03d}.h5'.format(idx + 1)

    return history, best_model_name

def test_execution(experiment, group='test', load_model_name='', sleep_metrics=False, type='semantic'):

    directories = experiment.get_directories()
    data_plots = experiment.get_data_plot(group)
    model = experiment.get_model(load_model_name=load_model_name)
    train_parameters = experiment.get_train_params() # TODO - change this to get_params(group)
    thresholds_best, merge_size, min_len = post_parameter_search(model=model,
                                                                 test_dataset=experiment.get_group_dataset('validation')[0], # TODO - figure out what validation datasets you should use for this
                                                                 metric_to_optimize=train_parameters['monitor'],
                                                                 type=type)
    for n, data_plot in enumerate(data_plots):

        predict_params = {
            'model': model,
            'test_dataset': data_plot['database'],
            'thresholds': thresholds_best,
            'merge_size': merge_size,
            'min_len': min_len
        }

        if type == 'semantic':
            predict_params.update({'sleep_metrics': sleep_metrics, 'analysis_dataset': data_plot['analysis_database']})
            metrics_test, output_all, sleep_params = compute_semantic_performance_subject(**predict_params)
            data_plots[n]['sleep_parameters'].update(sleep_params)

        else:
            predict_params.update({'min_iou': 0.1,
                                   # 'prediction_explanable_events': [{'h5_path': 'PLM'}, {'h5_path': 'arousal'}]
                                   'prediction_explanable_events': [{'h5_path': 'arousal'}]})
            metrics_test = compute_metrics_subject(**predict_params)

        # TODO - we could make overall...
        for event in metrics_test.keys():
            data_plots[n]['performance'][event].update(metrics_test[event])
            if type == 'semantic':
                data_plots[n]['output_all'][event].update(output_all[event])
        data_plot['database'] = ''
    return data_plots


