from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.models import Model
from ..blocks import add_common_layers_1D, RNN
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.regularizers import l2

def recurrent_model(input_shape,
                    number_of_classes,
                    num_outputs,
                    output_layer='sigmoid',
                    last_dropout_rate=0,
                    init_filter_num=64,
                    dropout_rate=0,
                    BN_momentum=.95,
                    layer_depth=3,
                    weight_regularization=0,
                    input_dropout_rate=0,
                    output_dropout_rate=0):

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Dropout(input_dropout_rate)(x)

    # Sequential model
    x = rnn_block(x,
                  init_filter_num=init_filter_num,
                  dropout_rate=dropout_rate,
                  BN_momentum=BN_momentum,
                  layer_depth=layer_depth,
                  weight_regularization=weight_regularization)

    # Classification
    x = layers.Dropout(output_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    return Model(inputs=x_input, outputs=x)


def rnn_block(x, init_filter_num=64, dropout_rate=0, BN_momentum=.95, layer_depth=1, weight_regularization=0):
    for layer_num in range(layer_depth):
        x = RNN(x, init_filter_num=init_filter_num, weight_regularization=weight_regularization)
        x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
    return x
