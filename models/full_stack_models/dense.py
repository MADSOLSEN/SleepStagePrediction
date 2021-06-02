from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.models import Model
from ..blocks import add_common_layers_1D, RNN
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.regularizers import l2

def dense_model(input_shape,
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
    init_filter_num = 32
    filter_increment_factor = 2
    depth = determine_depth(input_shape[0], num_outputs, max_pool_size[0])

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    if input_mode == '1D':
        x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)
        x = zeropad_1D(x, num_outputs=num_outputs, num_halfs=depth * max_pool_size[0] // 2)
    else:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2], 1))(x)  # to 2D:
        x = zeropad_2D(x, num_outputs=num_outputs, num_halfs=depth * max_pool_size[0] // 2)

    x = cnn_block(x,
                  init_filter_num=init_filter_num,
                  depth=depth,
                  filter_increment_factor=filter_increment_factor,
                  kernel_size=kernel_size,
                  max_pool_size=max_pool_size,
                  dropout=dropout,
                  layer_normalization=layer_normalization,
                  activation=activation,
                  weight_decay=weight_decay,
                  data_format=data_format)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)


    # Sequential model
    x = dense(x,
              init_filter_num=512,
              filter_increase_factor=filter_increment_factor,
              dropout_rate=dropout,
              layer_depth=1,
              num_layers=1,
              maxpool=False)


    # Classification
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    return Model(inputs=x_input, outputs=x)


def dense(x, init_filter_num, filter_increase_factor, dropout_rate,
          layer_depth, num_layers, maxpool=False, add_common_layers=True, BN_momentum=0.95):

    for n in range(num_layers):
        for m in range(layer_depth):
            x = layers.Dense(units=init_filter_num * filter_increase_factor ** (n),
                             kernel_initializer='he_normal')(x)
            if add_common_layers:
                x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
        if maxpool:
            x = layers.MaxPooling1D(pool_size=2)(x)
    return x


def cnn_block(x, init_filter_num, depth, filter_increment_factor, kernel_size, max_pool_size, dropout,
              layer_normalization, activation, weight_decay, data_format):

    features = init_filter_num
    for i in range(depth):
        skip = x
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

        # add dropout
        skip = layers.Conv2D(features, kernel_size=(1, 1), activation=activation, padding='same', data_format=data_format)(skip)
        x = layers.Add()([x, skip])
        x = layers.MaxPooling2D(max_pool_size, data_format=data_format)(x)
        features = features * filter_increment_factor

    return x

def zeropad_1D(x, num_outputs, num_halfs):
    if num_halfs < 0:
        return x
    adjusted_input_size = num_outputs * 2 ** num_halfs
    if adjusted_input_size != x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        #x = layers.ZeroPadding1D(padding=zeros_to_add)(x)
        x = layers.ZeroPadding2D(padding=(zeros_to_add, 0))(x)  # only padding the time axis!
    return x

def zeropad_2D(x, num_outputs, num_halfs):
    adjusted_input_size = num_outputs * 2 ** num_halfs
    if not (adjusted_input_size >= x.shape[1]):
        k = 1
    assert (adjusted_input_size >= x.shape[1])
    if adjusted_input_size > x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding2D(padding=(zeros_to_add, 0))(x) # only padding the time axis!

    return x

def determine_depth(temporal_shape_in, temporal_shape_out, temporal_max_pool_size):

    depth = 0
    temp_temporal_size = temporal_shape_out
    while temp_temporal_size < temporal_shape_in:
        depth += 1
        temp_temporal_size *= temporal_max_pool_size

    return depth