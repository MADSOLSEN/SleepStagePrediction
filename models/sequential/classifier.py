from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from ..blocks import add_common_layers_1D
from tensorflow.keras.regularizers import l2

def classifier_model(input_shape,
                     number_of_classes,
                     num_outputs,
                     output_layer='sigmoid',
                     add_dense=False,
                     init_filter_num=128,
                     filter_increase_factor=1,
                     num_layers=1,
                     layer_depth=1,
                     dropout_rate=0,
                     BN_momentum=0.95,
                     maxpool=True,
                     weight_regularization=0,
                     input_dropout_rate=0,
                     output_dropout_rate=0):

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    if input_dropout_rate > 0:
        x = layers.Dropout(input_dropout_rate)(x)

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)

    if add_dense:
        x = dense(x,
                  init_filter_num=init_filter_num,
                  filter_increase_factor=filter_increase_factor,
                  dropout_rate=dropout_rate,
                  layer_depth=layer_depth,
                  BN_momentum=BN_momentum,
                  num_layers=num_layers,
                  maxpool=maxpool,
                  weight_regularization=weight_regularization)

    if output_dropout_rate > 0:
        x = layers.Dropout(output_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    model = Model(inputs=x_input, outputs=x)
    return model

def dense(x, init_filter_num=16, filter_increase_factor=1, dropout_rate=0.5, weight_regularization=0,
          layer_depth=2, num_layers=7, BN_momentum=0.95, maxpool=True):

    for n in range(num_layers):
        for m in range(layer_depth):
            x = layers.Dense(units=init_filter_num * filter_increase_factor ** (n + 1),
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_regularization),
                             bias_regularizer=l2(weight_regularization))(x)
            x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
        if maxpool:
            x = layers.MaxPooling1D(pool_size=2)(x)
    return x