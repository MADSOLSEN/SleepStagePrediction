from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2


def USleep(input_shape,
               number_of_classes,
               num_outputs,
               depth,
               weight_regularization=0,
               init_filter_num=32,
               filter_increment_factor=2 ** (1/2),
               kernel_size=(32, 1),
               max_pool_size=(4, 1),
               dropout=0.0,
               layer_normalization=True,
               activation='elu',
               softmax=True,
               final_cnn=True,
               data_format='channels_last'):
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)  # to 2D:

    features = init_filter_num
    skips = []

    # encoder
    for i in range(depth):
        x = layers.Conv2D(features,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          data_format=data_format,
                          kernel_regularizer=l2(weight_regularization),
                          bias_regularizer=l2(weight_regularization))(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)
        skips.append(x)
        x = layers.MaxPooling2D((max_pool_size, 1), data_format=data_format)(x)
        features = features * filter_increment_factor

    x = layers.Conv2D(features,
                      kernel_size=kernel_size,
                      activation=activation,
                      padding='same',
                      data_format=data_format,
                      kernel_regularizer=l2(weight_regularization),
                      bias_regularizer=l2(weight_regularization))(x)
    if layer_normalization:
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # decoder
    for i in reversed(range(depth)):
        features = np.ceil(features // filter_increment_factor)
        x = layers.Conv2D(features,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          data_format=data_format,
                          kernel_regularizer=l2(weight_regularization),
                          bias_regularizer=l2(weight_regularization))(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    if final_cnn:
        x = layers.Conv1D(number_of_classes,
                          kernel_size=1,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation=activation)(x)

    if softmax:
        x_sigmoid = layers.Dense(units=number_of_classes - 4,
                                 activation='sigmoid',
                                 kernel_initializer='he_normal')(x)
        x_softmax = layers.Dense(units=4,
                                 activation='softmax',
                                 kernel_initializer='he_normal')(x)
        x = layers.Concatenate(axis=-1)([x_sigmoid, x_softmax])
    else:
        x = layers.Dense(units=number_of_classes,
                         activation='sigmoid',
                         kernel_initializer='he_normal')(x)

    x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)

    return Model(inputs=x_input, outputs=x)
