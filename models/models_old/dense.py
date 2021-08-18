from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def dense_model(input_shape,
                batch_size,
                number_of_classes,
                init_filter_num,
                filter_increase_factor,
                cardinality,
                dropout_rate,
                kernel_size,
                layer_depth,
                num_layers_encoder,
                num_layers_decoder,
                num_outputs,
                last_dropout_rate=0,
                dilation_rate=1,
                BN_momentum=0.99,
                add_rnn=False,
                add_attention=False,
                add_classifier=False,
                output_layer='sigmoid',
                maxpool=True):

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    if maxpool:
        x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers_encoder - num_layers_decoder)

    # model
    x = dense(x,
              init_filter_num=init_filter_num,
              filter_increase_factor=filter_increase_factor,
              dropout_rate=dropout_rate,
              layer_depth=layer_depth,
              dilation_rate=dilation_rate,
              BN_momentum=BN_momentum,
              num_layers_encoder=num_layers_encoder,
              maxpool=maxpool)

    if add_rnn:
        if x.shape[1] != 1:
            x = RNN(x,
                    init_filter_num=64,
                    dropout_rate=dropout_rate,
                    BN_momentum=BN_momentum,
                    layer_depth=1)

    if add_classifier:
        x = layers.SpatialDropout1D(last_dropout_rate)(x)
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)

    return Model(inputs=x_input, outputs=x)

def zeropad(x, num_outputs, num_halfs):

    adjusted_input_size = num_outputs * 2 ** num_halfs
    if adjusted_input_size != x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding1D(padding=zeros_to_add)(x)
    return x

def RNN(x, init_filter_num=64, dropout_rate=0.5, layer_depth=1, BN_momentum=0.99):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Bidirectional(layers.GRU(init_filter_num,
                                            return_sequences=True,
                                            kernel_initializer="he_normal"))(x)

        x = add_common_layers(x)

    return x


def dense(x, init_filter_num=16, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0.5,
          layer_depth=2, num_layers_encoder=7, dilation_rate=1, BN_momentum=0.95, maxpool=True):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y
    k = 1
    for n in range(num_layers_encoder):
        for m in range(layer_depth):
            x = layers.Dense(units=init_filter_num * filter_increase_factor ** (n + 1),
                             kernel_initializer='he_normal')(x)
            x = add_common_layers(x)
        if maxpool:
            x = layers.MaxPooling1D(pool_size=2)(x)
    return x
