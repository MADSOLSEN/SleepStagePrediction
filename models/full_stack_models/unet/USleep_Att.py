from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K



def USleep_Att(input_shape,
               number_of_classes,
               num_outputs,
               output_layer='softmax',
               weight_decay=0.0,
               init_filter_num=16,
               filter_increment_factor=2 ** (1/2),
               kernel_size=(32, 1),
               max_pool_size=(4, 1),
               dropout=0.0,
               layer_normalization=True,
               activation='elu',
               attention=True,
               final_cnn=True,
               spatial_filtering=False,
               data_format='channels_last',
               input_mode='1D'):

    skips = []
    features = init_filter_num
    depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0])


    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    if input_mode == '1D':
        x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)
    else:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2], 1))(x)  # to 2D:


    if spatial_filtering:
        x_ = layers.Permute((3, 1, 2))(x)
        x_ = layers.Conv2D(x_.shape[1],
                           kernel_size=(x_.shape[1], 1),
                           strides=(1, 1),
                           kernel_initializer='he_normal')(x_)
        x = layers.Permute((2, 1, 3))(x_)


    # encoder
    for i in range(depth):

        x = layers.Conv2D(features,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          data_format=data_format,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay))(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)
        skips.append(x)
        x = layers.MaxPooling2D(max_pool_size, data_format=data_format)(x)
        features = features * filter_increment_factor

    x = layers.Conv2D(features,
                      kernel_size=kernel_size,
                      activation=activation,
                      padding='same',
                      kernel_initializer='he_normal',
                      data_format=data_format,
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(dropout)(x)
    if layer_normalization:
        x = layers.LayerNormalization()(x)


    # decoder
    for i in reversed(range(depth)):
        features = features / filter_increment_factor

        if attention:
            x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size, data_format=data_format)
        else:
            x = layers.UpSampling2D(size=max_pool_size, data_format=data_format)(x)
            x = layers.Conv2D(features, max_pool_size, activation=activation, padding='same', data_format=data_format)(x)
            x = layers.concatenate([skips[i], x], axis=3)

        x = layers.Conv2D(int(features),
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          kernel_initializer='he_normal',
                          data_format=data_format,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay))(x)
        x = layers.Dropout(dropout)(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    if final_cnn:
        x = layers.Conv1D(filters=init_filter_num,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)

    if input_shape[0] // num_outputs > 0:
        x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)

    if final_cnn:
        x = layers.Conv1D(filters=number_of_classes,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)


    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)



    return Model(inputs=x_input, outputs=x)

def determine_depth(temporal_shape, temporal_max_pool_size):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= temporal_max_pool_size
    depth -= 1
    return depth