from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from models.blocks import add_common_layers_1D, conv_layer_1D, residual_block_1D, wavenet_residual_block_1D
import numpy as np
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from models import models


def build_model(input_shape,
                number_of_classes,
                num_outputs,
                input_prep,
                model,
                output_prep,

                kernel_size,
                num_layers_encoder,
                num_layers_decoder,
                init_filter_num=32,
                filter_increase_factor=2,
                max_pool_size=2,
                dropout_rate=0,
                layer_depth=1,
                cardinality=1,
                bottle_neck=False,
                dilation_rate=1,
                BN_momentum=0.95,
                input_dropout=False,
                output_layer='sigmoid',
                block_type='conv_1D',
                weight_regularization=0,
                input_dropout_rate=0,
                output_dropout_rate=0.5,
                sridhar_local_initiation=False,
                attention=False):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    # input
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Dropout(input_dropout_rate)(x)

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)

    # adjust time axis:
    adjust_size = False
    adjusted_temporal_input_size = input_shape[0]
    if adjust_size:
        adjusted_temporal_input_size = int(2 ** np.ceil(np.log2(input_shape[0])))
        padding = (adjusted_temporal_input_size - input_shape[0]) // 2
        x = layers.ZeroPadding1D(padding=padding)(x)

    #
    if sridhar_local_initiation:
        # TODO - Must be updated!
        block_size = 30 * 4
        block_stride = 30
        fs = 4
        x = overlapping_blocker(x, block_size=block_size * fs, stride=block_stride * fs)

        x = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same')(x)

        # Local convolutional part
        for layer_num in range(4):
            skip = layers.MaxPool1D(pool_size=2, strides=2)(x)
            for depth_num in range(2):
                x = layers.Conv1D(filters=16, kernel_size=6, strides=1, padding='same')(x)
                x = layers.LeakyReLU(alpha=0.15)(x)
            x = layers.MaxPool1D(pool_size=2, strides=2)(x)
            x = layers.Add()([x, skip])

        # x = tf.reshape(x, [-1, num_outputs, x.shape[1] * x.shape[2]])
        x = tf.reshape(x, [-1, num_outputs, x.shape[1] * x.shape[2]])
        num_layers_encoder = 2
        num_layers_decoder = 2
        init_filter_num = 128


    # model
    models

    # classifier
    x = layers.Dropout(output_dropout_rate)(x)
    x = layers.AveragePooling1D(pool_size=adjusted_temporal_input_size // num_outputs)(x)

    # add attention here!
    #x = layers.MultiHeadAttention(num_heads=3, key_dim=2)(x)
    #x = layers.MultiHeadAttention(num_heads=3, key_dim=2, )(x, x, return_attention_scores=False)

    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)

    return Model(inputs=x_input, outputs=x)