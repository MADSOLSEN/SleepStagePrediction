from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from ..blocks import add_common_layers_1D


def dense_model(input_shape,
                num_layers,
                num_outputs,
                init_filter_num,
                kernel_size=3,
                filter_increase_factor=1,
                layer_depth=1,
                dropout_rate=0,
                BN_momentum=0.95,
                maxpool=True,
                add_common_layers=True,
                block_type='',
                bottle_neck='',
                cardinality='',
                chunk_time='',
                weight_regularization=0):

    # input
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    if maxpool:
        x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    # model
    x = dense(x,
              init_filter_num=init_filter_num,
              filter_increase_factor=filter_increase_factor,
              dropout_rate=dropout_rate,
              layer_depth=layer_depth,
              BN_momentum=BN_momentum,
              num_layers=num_layers,
              maxpool=maxpool,
              add_common_layers=add_common_layers)

    return Model(inputs=x_input, outputs=x)

def zeropad(x, num_outputs, num_halfs):

    adjusted_input_size = num_outputs * 2 ** num_halfs
    if adjusted_input_size != x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding1D(padding=zeros_to_add)(x)
    return x


def dense(x, init_filter_num=16, filter_increase_factor=1, dropout_rate=0.5,
          layer_depth=2, num_layers=7, BN_momentum=0.95, maxpool=True, add_common_layers=True):

    for n in range(num_layers):
        for m in range(layer_depth):
            x = layers.Dense(units=init_filter_num * filter_increase_factor ** (n + 1),
                             kernel_initializer='he_normal')(x)
            if add_common_layers:
                x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
        if maxpool:
            x = layers.MaxPooling1D(pool_size=2)(x)
    return x
