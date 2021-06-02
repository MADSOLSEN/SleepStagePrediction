from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def stan_model(input_shape,
               number_of_classes,
               init_filter_num,
               filter_num_iterations,
               cardinality,
               dropout_rate,
               kernel_size,
               layer_depth,
               num_layers_encoder,
               num_layers_decoder,
               num_outputs,
               dilation_rate=1,
               BN_momentum=0.99,
               add_rnn=False,
               add_classifier=True,
               output_layer='sigmoid',
               input_dropout=False):

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers_encoder - num_layers_decoder)

    # model
    x = stan(x,
             init_filter_num=8,
             kernel_size=kernel_size,
             dropout_rate=dropout_rate,
             BN_momentum=BN_momentum,
             num_layers=num_layers_encoder - num_layers_decoder)

    if add_rnn:
        x = RNN(x,
                init_filter_num=64,
                dropout_rate=dropout_rate,
                BN_momentum=BN_momentum,
                layer_depth=1)

    if add_classifier:
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)

    return Model(inputs=x_input, outputs=x)


def zeropad(x, num_outputs, num_halfs):
    if num_halfs < 0:
        return x
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


def stan(x, init_filter_num=8, kernel_size=16, dropout_rate=0.5, num_layers=5, BN_momentum=0.95):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        return y

    # spatial filtering:
    x_ = layers.Permute((2, 1))(x)
    x_ = layers.Reshape((x_.shape[1], x_.shape[2], 1))(x_)
    x_ = layers.Conv2D(x_.shape[1],
                       kernel_size=(x_.shape[1], 1),
                       strides=(1, 1),
                       kernel_initializer='he_normal')(x_)
    x_ = layers.Permute((3, 2, 1))(x_)
    layer_counter = 1

    for n in range(num_layers):
        x_ = layers.Conv2D(init_filter_num * layer_counter,
                           kernel_size=(1, 16),
                           strides=(1, 1),
                           padding='same',
                           kernel_initializer='he_normal')(x_)
        x_ = add_common_layers(x_)
        x_ = layers.MaxPooling2D(pool_size=(1, 2))(x_)
        layer_counter += 1
    x_ = layers.Permute((2, 1, 3))(x_)
    x_ = layers.Reshape((x_.shape[1], x_.shape[2] * x_.shape[3]))(x_)
    x_ = layers.SpatialDropout1D(.5)(x_)
    return x_
