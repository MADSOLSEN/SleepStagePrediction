from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from ..blocks import add_common_layers_1D, conv_layer_1D, residual_block_1D, wavenet_residual_block_1D
import numpy as np
from tensorflow.keras.regularizers import l2

def papini_model(input_shape,
                 number_of_classes,
                 num_outputs,
                 filter_increase_factor=2,
                 output_layer='sigmoid',
                 weight_regularization=0,
                 input_dropout_rate=0,
                 output_dropout_rate=0):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Dropout(input_dropout_rate)(x)

    # model
    x = papini(x, weight_regularization=weight_regularization)

    x = layers.Dropout(output_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    return Model(inputs=x_input, outputs=x)


def papini(x, weight_regularization=0):


    # block 1
    x = layers.Dropout(.2)(x)
    x = layers.GaussianNoise(stddev=0.01)(x)

    # block type 3
    for n in range(2):
        x = layers.Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(stddev=0.01)(x)

        x = layers.Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(stddev=0.01)(x)

        x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.GaussianNoise(stddev=0.01)(x)

    # block type 2
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GaussianNoise(stddev=0.01)(x)

    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GaussianNoise(stddev=0.01)(x)

    # output block
    x = layers.Dense(units=16)(x)
    x = layers.Dropout(.3)(x)

    return x
