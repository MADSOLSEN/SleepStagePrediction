import os
import numpy as np
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import concatenate
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
import tensorflow as tf


from functions import loss_functions, metric_functions, metrics
from signal_processing import post_processing
from utils import binary_to_array, non_max_suppression, intersection_overlap


def predict_dataset(model,
                    dataset,
                    threshold,
                    merge_size,
                    min_len):
    
    # get it in the following format:
    events = [event['name'] for event in dataset.events_format]
    predictions = {
        record: {
            event: []
            for event in events
        } for record in dataset.records
    }
    for record in dataset.records:
        localizations, scores = {event: [] for event in events}, {event: [] for event in events}

        for signals, start_indexes in dataset.get_record_batch(record):
            predictions_batch = model.predict_on_batch(signals)

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
                            overlap = intersection_overlap(np.array([ed_it['data'][0, :] // dataset.fs,
                                                                     (ed_it['data'][0, :] + ed_it['data'][1, :]) // dataset.fs]).T,
                                                           np.array(predictions_))
                            if len(overlap) > 0:
                                max_iou = overlap.max(0)
                                keep_bool = (max_iou < dataset.discard_threshold)
                                predictions_ = [pred for pred, keep_bool in zip(predictions_, keep_bool) if keep_bool]

                    for es_key, es_it in dataset.events_select[record].items():
                        if predictions_:
                            overlap = intersection_overlap(np.array([es_it['data'][0, :] // dataset.fs,
                                                                     (es_it['data'][0, :] + es_it['data'][1, :]) // dataset.fs]).T,
                                                           np.array(predictions_))
                            if len(overlap) > 0:
                                max_iou = overlap.max(0)
                                keep_bool = (max_iou > (dataset.select_threshold))
                                predictions_ = [pred for pred, keep_bool in zip(predictions_, keep_bool) if keep_bool]

                predictions[record][event] = predictions_

    return predictions

def predict_dataset_semantic(model,
                             dataset):

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
                event_ = dataset.get_events(record=record, index=int(start_index))
                localizations_ = list(range(int(start_index) // dataset.prediction_resolution,
                                            int(start_index) // dataset.prediction_resolution + dataset.predictions_per_window))
                for loc_ in localizations_:
                    if loc_ > localizations[-1]:
                        localizations += [loc_]
                        for class_num, event in enumerate(events):
                            predictions[record][event] += [prediction_window[loc_ % dataset.predictions_per_window, class_num]]
                            targets[record][event] += [event_[loc_ % dataset.predictions_per_window, class_num]]
        max_index = [idr['max_index'] for idr in dataset.index_to_record if idr['record'] == record][0]
        assert(len(localizations) - 1 == int((max_index + dataset.window) / (dataset.prediction_resolution)))

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
          patience,
          initial_epoch,
          model_name,
          model_directory,
          tensorboard_directory,
          training_log_directory,
          loss,
          monitor,
          learning_rate):

    # step 1 - compile
    model.compile(loss=loss_functions[loss['type']](**loss['args']),
                  optimizer=Adam(lr=learning_rate, epsilon=1e-8),
                  metrics=[metrics['binary_accuracy']()] +
                          [metrics['balanced_accuracy'](num_classes=train_dataset.number_of_classes)] +
                          [metrics['balanced_accuracy_new'](num_classes=train_dataset.number_of_classes)] +
                          [metrics['f1'](beta=1)] +
                          [metrics['precision']()] +
                          [metrics['recall']()] +
                          [metrics['cohens_kappa'](num_classes=train_dataset.number_of_classes)] #+
                          #[metrics['cohens_kappa1'](num_classes=train_dataset.number_of_classes, lower_bound=.25, upper_bound=.75, num_samples=50)] +
                          #[metrics['optimized_f1'](lower_bound=.25, upper_bound=.75, num_samples=50)] +
                          #[metrics['optimized_f12'](num_classes=train_dataset.number_of_classes, lower_bound=.25, upper_bound=.75, num_samples=50)]  # +
                  #[metrics['MCC']] +
                  #[metrics['AUCRC'](lower_bound=.01, upper_bound=.99, num_samples=100)] +
                  #[metrics['AUROC'](lower_bound=.01, upper_bound=.99, num_samples=100)]
                  )

    # step 2 - get callbacks
    callbacks = get_callbacks(model_name=model_name,
                              monitor=monitor,
                              patience=patience,
                              model_directory=model_directory,
                              tensorboard_directory=tensorboard_directory,
                              training_log_directory=training_log_directory)
    k = 1
    # step 3 - fit generator
    # tf.compat.v1.disable_eager_execution()
    history = model.fit_generator(train_dataset,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=2,
                                  initial_epoch=initial_epoch,
                                  validation_data=validate_dataset)

    return history


def get_callbacks(model_name,
                  monitor,
                  patience,
                  model_directory, 
                  tensorboard_directory, 
                  training_log_directory):
    
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
                                  patience=13)
    lr_reducer = ReduceLROnPlateau(monitor='val_' + monitor,
                                   mode='max',
                                   factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=5e-7,
                                   verbose=1)

    # tensorboard
    tensorboard = TensorBoard(tensorboard_directory,
                              write_graph=True,
                              write_images=True)

    # training log
    csv_logger = CSVLogger(os.path.join(training_log_directory, 'training.log'))

    return  [checkpoint, earlystopping, lr_reducer, tensorboard, csv_logger]


def reconfigure_model(model, num_layers_pop, freeze_all_but_last):
    # Build pretrained model
    for n in range(num_layers_pop):
        model.layers.pop()

    # Freeze layers
    if freeze_all_but_last == 0:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:-freeze_all_but_last]:
            layer.trainable = False

    model = Model(inputs=model.input,
                  outputs=model.layers[-1].output)


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
