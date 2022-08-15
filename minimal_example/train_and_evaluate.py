
data_directory = ''


train_arc, validation_arc, test_arc = get_train_validation_test(directories['arc'],
                                                                        percent_test=50,
                                                                        percent_validation=12.5,
                                                                        seed=self.seed)


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


