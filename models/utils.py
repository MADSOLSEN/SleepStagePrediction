import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import AdamW

# from lingvo.core import optimizer

import tensorflow as tf
import wandb
import h5py
#from wandb.keras import WandbCallback




from functions import loss_functions, metrics, learning_rate_schedules
from signal_processing import post_processing
from utils import binary_to_array, non_max_suppression, intersection_overlap, create_directory_tree


def predict_dataset(model,
                    dataset,
                    threshold,
                    merge_size,
                    min_len,
                    events):
    
    # get it in the following format:
    # events = [event['name'] for event in datasets.events_format]
    predictions = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }
    for record in dataset.records:
        localizations, scores = {event: [] for event in events}, {event: [] for event in events}
        predictions_batch = []
        for signals, start_indexes in dataset.get_record_batch(record):
            for n in range(len(start_indexes)):
                signal = [s[n:n+1, :] for s in signals]
                predictions_batch += [model.predict_on_batch(signal)[0, :]]
            predictions_batch = np.array(predictions_batch)
            # predictions_batch = model.predict_on_batch(signals)

            for prediction_window, start_index in zip(predictions_batch, start_indexes):
                for class_num, event in enumerate(events):
                    prediction_binary = prediction_window[:, class_num] > threshold[event]          # apply threshold
                    pp_prediction_binary = post_processing(prediction_binary,                       # post processing
                                                           merge_size=merge_size[event],
                                                           min_len=min_len[event])
                    localization_relative = binary_to_array(pp_prediction_binary)                   # get start stop.

                    for loc in localization_relative:                                               # only activated if any events present.
                        scores[event] += [np.mean(prediction_window[loc[0]:loc[1], class_num])]     # find score for each event
                        localizations[event] += [[l * dataset.prediction_resolution + start_index for l in loc]]

        for event in events:
            if localizations[event]:
                predictions_ = non_max_suppression(np.array(localizations[event]),
                                                   np.array(scores[event]),
                                                   overlap=0.5)

                if dataset.use_mask:
                    for ed_key, ed_it in dataset.events_discard[record].items():
                        if predictions_:
                            overlap = intersection_overlap(np.array([ed_it['data'][0, :] // dataset.prediction_resolution ,
                                                                     (ed_it['data'][0, :] + ed_it['data'][1, :]) // dataset.prediction_resolution]).T,
                                                           np.array(predictions_))
                            if len(overlap) > 0:
                                max_iou = overlap.max(0)
                                keep_bool = (max_iou < dataset.discard_threshold)
                                predictions_ = [pred for pred, keep_bool in zip(predictions_, keep_bool) if keep_bool]

                    for es_key, es_it in dataset.events_select[record].items():
                        if predictions_:
                            overlap = intersection_overlap(np.array([es_it['data'][0, :] // dataset.prediction_resolution,
                                                                     (es_it['data'][0, :] + es_it['data'][1, :]) // dataset.prediction_resolution]).T,
                                                           np.array(predictions_))
                            if len(overlap) > 0:
                                max_iou = overlap.max(0)
                                keep_bool = (max_iou > (dataset.select_threshold))
                                predictions_ = [pred for pred, keep_bool in zip(predictions_, keep_bool) if keep_bool]

                predictions[record][event] = predictions_
    keras.backend.clear_session()
    return predictions

def predict_dataset_semantic(model, dataset, save_prediction_path=''):

    # get it in the following format:
    no_event = [] #[{'name': 'no_event', 'h5_path': 'no_event', 'probability': .5}]
    events = [event['name'] for event in (dataset.events_format + no_event)]

    predictions = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }
    targets = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }


    if len(save_prediction_path) > 0:

        # new_dir = os.path.join(datasets.h5_directory, 'model_output')

        new_dir = os.path.join(save_prediction_path, 'model_output')
        create_directory_tree(new_dir)

    for record in dataset.records:
        if record == '18-01141.h5':
            k = 1
        localizations = [-1]
        predictions_batch = []
        for signals, start_indexes in dataset.get_record_batch(record):
            for n in range(len(start_indexes)):
                signal = [s[n:n+1, :] for s in signals]
                predictions_batch += [model.predict_on_batch(signal)[0, :]]
            predictions_batch = np.array(predictions_batch)
            for prediction_window, start_index in zip(predictions_batch, start_indexes):
                event_ = dataset.get_events(record=record, index=int(start_index))
                localizations_ = list(range(int(start_index) // dataset.prediction_resolution,
                                            int(start_index) // dataset.prediction_resolution + dataset.predictions_per_window))
                for num, loc_ in enumerate(localizations_):
                    if loc_ > localizations[-1]:
                        localizations += [loc_]

                        for class_num, event in enumerate(events):
                            predictions[record][event] += [prediction_window[num, class_num]]
                            targets[record][event] += [event_[num, class_num]]
        del localizations[0]
        max_index = [idr['max_index'] for idr in dataset.index_to_record if idr['record'] == record][0]
        #if len(localizations) % 5 > 0:

        #    print('FUUUUUUUUUUUUCK ')
        #    print(record)
        #    print(len(localizations))
        #    print(start_indexes)
        #    print(max_index)

        assert(len(localizations) == int((max_index + dataset.window) / (dataset.prediction_resolution)))
        # assert(len(localizations) == predictions)

        if len(save_prediction_path) > 0:
            h5_full_filename = os.path.join(new_dir, record)

            with h5py.File(h5_full_filename, 'w') as h5:
                for name, item in predictions[record].items():
                    h5.create_dataset(name, data=item)
                    h5[name].attrs["fs"] = 1 / dataset.prediction_resolution

    keras.backend.clear_session()
    return predictions, targets


def extract_analysis_feature(dataset):

    # get it in the following format:
    no_event = [] #[{'name': 'no_event', 'h5_path': 'no_event', 'probability': .5}]
    events = [event['name'] for event in (dataset.events_format + no_event)]

    predictions = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }
    targets = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }

    for record in dataset.records:
        localizations = [-1]
        for signals, start_indexes in dataset.get_record_batch(record):
            predictions_batch = model.predict_on_batch(signals)
            for prediction_window, start_index in zip(predictions_batch, start_indexes):
                signal_, event_ = dataset.get_sample(record=record, index=int(start_index * dataset.fs))
                #print(start_index)
                localizations_ = list(range(int(start_index) // dataset.prediction_resolution,
                                            int(start_index) // dataset.prediction_resolution + dataset.predictions_per_window))
                for loc_ in localizations_:
                    if loc_ > localizations[-1]:
                        localizations += [loc_]
                        for class_num, event in enumerate(events):
                            predictions[record][event] += [prediction_window[loc_ % dataset.predictions_per_window, class_num]]
                            targets[record][event] += [event_[loc_ % dataset.predictions_per_window, class_num]]
        max_index = [idr['max_index'] for idr in dataset.index_to_record if idr['record'] == record][0]
        assert(len(localizations) - 1 == int((max_index + dataset.window_size) / (dataset.fs * dataset.prediction_resolution)))

    return predictions, targets


def train(model,
          train_dataset,
          validate_dataset,
          epochs,
          initial_epoch,
          model_name,
          model_directory,
          tensorboard_directory,
          training_log_directory,
          load_model_name,
          loss,
          monitor,
          learning_rate,
          logging_dict={},
          patience=18,
          reduce_patience=6):

    from wandb.keras import WandbCallback
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
    K.clear_session()
    # step 1 - compile


    model.compile(loss=loss_functions[loss['type']](**loss['args']),
                  run_eagerly=False,
                  optimizer=Adam(learning_rate=learning_rate, epsilon=1e-8),# optimizer.DistributedShampoo(learning_rate=learning_rate),
                  #optimizer=AdamW(learning_rate=learning_rate, weight_decay=0.002, epsilon=1e-8), # optimizer.DistributedShampoo(learning_rate=learning_rate),
                  metrics=[metrics['binary_accuracy']()] +
                          [metrics['balanced_accuracy'](num_classes=train_dataset.number_of_classes)] +
                          [metrics['balanced_accuracy_new'](num_classes=train_dataset.number_of_classes)] +
                          [metrics['f1_mean'](beta=1)] +
                          [metrics['precision']()] +
                          [metrics['recall']()] +
                          [metrics['cohens_kappa'](num_classes=train_dataset.number_of_classes)]
                          #[metrics['f1_by_class'](class_idx=0)], # for class_idx in range(train_dataset.number_of_classes)]
                  #[metrics['cohens_kappa1'](num_classes=train_dataset.number_of_classes, lower_bound=.25, upper_bound=.75, num_samples=50)] +
                  #[metrics['optimized_f1'](lower_bound=.25, upper_bound=.75, num_samples=50)] +
                  #[metrics['optimized_f12'](num_classes=train_dataset.number_of_classes, lower_bound=.25, upper_bound=.75, num_samples=50)]  # +
                  #[metrics['MCC']] +
                  #[metrics['AUCRC'](lower_bound=.01, upper_bound=.99, num_samples=100)] +
                  #[metrics['AUROC'](lower_bound=.01, upper_bound=.99, num_samples=100)]
                  #config=run_opts,
                  #metrics={"dense": {'bin_acc': metrics['binary_accuracy'](),
                  #                   'balanced_accuracy': metrics['balanced_accuracy'](num_classes=train_dataset.number_of_classes),
                  #                   'f1/wake': metrics['f1_by_class'](class_idx=0),
                  #                   'f1/light': metrics['f1_by_class'](class_idx=1)}}
                           # 'balanced_accuracy': metrics['balanced_accuracy'(num_classes=train_dataset.number_of_classes)#,
                           #v
                  #         }

                          #[metrics['balanced_accuracy'](num_classes=train_dataset.number_of_classes)] +
                          #[metrics['balanced_accuracy_new'](num_classes=train_dataset.number_of_classes)] +
                          #[metrics['f1_mean'](beta=1)] +
                          #[metrics['precision']()] +
                          #[metrics['recall']()] +
                          #[metrics['cohens_kappa'](num_classes=train_dataset.number_of_classes)] +
                          #[metrics['f1_by_class'](class_idx=0)]
                  )

    # step 2 - get callbacks
    callbacks = get_callbacks(model_name=model_name,
                              monitor=monitor,
                              patience=patience,
                              reduce_patience=reduce_patience,
                              model_directory=model_directory,
                              tensorboard_directory=tensorboard_directory,
                              learning_rate=learning_rate)
                              #training_log_directory=training_log_directory)

    model_dir_short = model_directory.split("\\")

    # TODO - Change this!
    project = 'SDB-detection' if model_dir_short[3] == 'SDB' else 'SleepStagePrediction'

    # weights and biases:
    wandb.init(project=project,#'SleepStagePrediction', #'SDB-detection',
               config=logging_dict,
               entity='madsol',
               sync_tensorboard=True,
               group=model_dir_short[4],
               name='/'.join(model_dir_short[4:])
               )

    callbacks += [WandbCallback(),
                  ClassMetrics(train_generator=train_dataset, val_generator=validate_dataset),
                  DatasetMetrics(train_generator=train_dataset, val_generator=validate_dataset)]

    # step 3 - fit generator
    history = model.fit(train_dataset,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                        initial_epoch=initial_epoch,
                        validation_data=validate_dataset)

    keras.backend.clear_session()
    wandb.finish()

    return history


def get_callbacks(model_name,
                  monitor,
                  patience,
                  reduce_patience,
                  model_directory, 
                  tensorboard_directory,
                  learning_rate=1e-3):#,
                  #training_log_directory):
    
    # save model
    filepath = os.path.join(model_directory, model_name + '.{epoch:03d}.h5')
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_' + monitor, # val_MCC # val_f1_score
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)

    # learning rate
    earlystopping = EarlyStopping(monitor='val_' + monitor,
                                  mode='max',
                                  min_delta=0,
                                  patience=patience)

    lr_schedule = ReduceLROnPlateau(monitor='val_' + monitor,
                                    mode='max',
                                    factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=reduce_patience,
                                    min_lr=1e-7,
                                    verbose=1)
    """
    lr_schedule = LearningRateScheduler(schedule=learning_rate_schedules['oscillator_exp_decay'](lr_min=1e-5,
                                                                                                 lr_max=learning_rate,
                                                                                                 lr_phase=10,
                                                                                                 warmup=patience - 5),
                                        verbose=0)
    """
    # tensorboard
    tensorboard = TensorBoard(tensorboard_directory,
                              write_graph=True,
                              write_images=True,
                              histogram_freq=10,
                              profile_batch=100)

    return  [checkpoint, earlystopping, lr_schedule, tensorboard]

class DatasetMetrics(Callback):

    def __init__(self, train_generator=None, val_generator=None):
        super().__init__()
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.max_train_steps = 50
        self.epoch_freq = 5
        self.dataset_names = {v: k for k, v in enumerate(train_generator.dataset_names)}

    def collect_in_batch(self, dataset_name, index_to_record, batch_size):

        dataset_names, records, indexes = [], [], []
        for count, idx_to_rec in enumerate(index_to_record):
            dataset_names.append(dataset_name)
            records.append(idx_to_rec['record'])
            indexes.append(idx_to_rec['index'])
            if (count + 1) % batch_size == 0:
                yield dataset_names, records, indexes
                dataset_names, records, indexes = [], [], []

    def on_epoch_end(self, epoch, logs=None):

        k = 1
        if epoch % self.epoch_freq == 0:
            # training
            train_dict = {}
            for dataset_name, dataset_idx in self.dataset_names.items():
                f1_rows = []
                assert(len(self.train_generator.index_to_record[dataset_name]) > self.train_generator.batch_size)
                for count, (dataset_names, records, indexes) in enumerate(self.collect_in_batch(dataset_name=dataset_name,
                                                                             index_to_record=self.train_generator.index_to_record[dataset_name],
                                                                             batch_size=self.train_generator.batch_size)):

                    train_data, train_labels = self.train_generator.final_prep(dataset_names, records, indexes)

                    y_pred = self.model.predict(train_data)
                    f1_macro = metrics['f1_mean'](beta=1)
                    f1_rows += [f1_macro(y_true=train_labels, y_pred=y_pred)]
                    if count > self.max_train_steps:
                        break
                train_dict['{}/{}'.format('train_f1', dataset_name)] = sum(f1_rows) / len(f1_rows)
            wandb.log(train_dict)

            # validation
            validation_dict = {}
            for dataset_name, dataset_idx in self.dataset_names.items():
                f1_rows = []
                for dataset_names, records, indexes in self.collect_in_batch(dataset_name=dataset_name,
                                                                             index_to_record=self.val_generator.index_to_record[dataset_name],
                                                                             batch_size=self.val_generator.batch_size):

                    val_data, val_labels = self.val_generator.final_prep(dataset_names, records, indexes)

                    y_pred = self.model.predict(val_data)
                    f1_macro = metrics['f1_mean'](beta=1)
                    f1_rows += [f1_macro(y_true=val_labels, y_pred=y_pred)]

                validation_dict['{}/{}'.format('val_f1', dataset_name)] = sum(f1_rows) / (len(f1_rows))
            wandb.log(validation_dict)



class ClassMetrics(Callback):
    """ Custom callback to compute per-class F1 at the end of each training epoch"""
    from collections import defaultdict

    def __init__(self, train_generator=None, val_generator=None):
        super().__init__()
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.epoch_freq = 5
        self.max_train_steps = min(50, train_generator.__len__())
        self.class_names = {v: k for k, v in enumerate(train_generator.event_labels)}

    def on_epoch_end(self, epoch, logs=None):

        # training:
        if epoch % self.epoch_freq == 0:
            f1_rows = []
            for i_ in range(self.max_train_steps):
                i = np.random.randint(0, self.train_generator.__len__())
                train_data, train_labels = self.train_generator[i]
                y_pred = self.model.predict(train_data)
                f1_rows += [self.log_f1_by_class(y_true=train_labels, y_pred=y_pred)]
            self.log_mean_of_classes(group_name='train_f1', rows_of_dict=f1_rows)

            f1_rows_val = []
            for val_data, val_labels in self.val_generator:
                y_pred = self.model.predict(val_data)
                f1_rows_val += [self.log_f1_by_class(y_true=val_labels, y_pred=y_pred)]
            self.log_mean_of_classes(group_name='val_f1', rows_of_dict=f1_rows_val)

     # TODO - define log_mean_of something else if you don't do classes.

    def log_mean_of_classes(self, group_name, rows_of_dict):
        temp_dict = {}
        for v, k in self.class_names.items():
            temp_dict['{}/{}'.format(group_name, v)] = sum([row[v] for row in rows_of_dict]) / len(rows_of_dict)
        wandb.log(temp_dict)

    def log_f1_by_class(self, y_true, y_pred):
        f1_dict = {}
        for v, k in self.class_names.items():
            f1_class_fun = metrics['f1_by_class'](class_idx=k)
            f1_dict[v] = f1_class_fun(y_true, y_pred)

        return f1_dict





def reconfigure_model(model, num_layers_pop, freeze_layers):

    if num_layers_pop > 0:
        input = model.input
        output = model.layers[-(num_layers_pop + 1)].output
        model = Model(inputs=input, outputs=output)

    if freeze_layers:
        for layer in model.layers:
            layer.trainable = False

    return model


def get_model_activation(model):
    # get layer outputs
    # TODO - maybe extract the output from batch_norm instead to get more stable results
    layer_outputs_rnn = [layer.output for layer in model.layers if layer.name[:4] == 'bidi']
    layer_outputs_cnn_temp = [layer.output for layer in model.layers if
                              layer.name[:4] == 'conv' and layer.input.name[:4] == 'spat']
    shape = layer_outputs_cnn_temp[0].shape[2]
    layer_outputs_cnn = []
    for layer in layer_outputs_cnn_temp:
        if layer.shape[2] == shape:
            last_layer = layer
        else:
            layer_outputs_cnn.append(last_layer)
            shape = layer.shape[2]

    layer_outputs = layer_outputs_cnn + layer_outputs_rnn
    # layer_outputs = [layer.output for layer in model.layers]

    # build activation model
    activation_model = Model(inputs=model.input,
                             outputs=layer_outputs)
    # Figure out how to sort the activations.
    return activation_model

def concatenate_models(models):
    model = keras.Sequential()
    for model_ in models:
        model.add(model_)
    return model

def stack_models(feature_extractors, classifier):
    input = feature_extractors.input
    output = classifier(feature_extractors.output)
    return Model(inputs=input, outputs=output)

def concatenate_model_outputs(models):
    inputs = [model.input for model in models]
    outputs = concatenate([model.output for model in models], axis=-1) if len(models) > 1 else models[0].output
    return Model(inputs=inputs, outputs=outputs)

def visualize_model(model, directory):
    plot_model(model, to_file=directory, show_shapes=True, show_layer_names=True)

def compute_receptive_field(desired_sample_rate, dilation_depth, nb_stacks):
    receptive_field = nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
    receptive_field_ms = (receptive_field * 1000) / desired_sample_rate
    return receptive_field, receptive_field_ms