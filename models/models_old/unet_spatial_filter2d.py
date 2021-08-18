from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def unet_spa_model(input_shape,
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
               add_rnn=True,
               add_attention=False,
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
    x = unet(x,
             init_filter_num=init_filter_num,
             filter_num_iterations=filter_num_iterations,
             kernel_size=kernel_size,
             cardinality=cardinality,
             dropout_rate=dropout_rate,
             layer_depth=layer_depth,
             dilation_rate=dilation_rate,
             BN_momentum=BN_momentum,
             num_layers_encoder=num_layers_encoder,
             num_layers_decoder=num_layers_decoder,
             input_dropout=input_dropout)

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


def unet2(x, init_filter_num=16, filter_num_iterations=16, kernel_size=3, cardinality=1, dropout_rate=0.5, layer_depth=2, num_layers_encoder=7,
         num_layers_decoder=2, dilation_rate=1, BN_momentum=0.95, input_dropout=False):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    def conv_layer(y, num_channels, kernel_size=3, strides=1, dilation_rate=1, data_format='channels_last'):

        if cardinality == 1:
            return layers.Conv1D(num_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer='he_normal')(y)

        assert not num_channels % cardinality
        _d_out = num_channels // cardinality
        _d_in = y.shape[-1] // cardinality
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, j * _d_in:j * _d_in + _d_in])(y)
            groups.append(layers.Conv1D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y


    d = {}
    # spatial filtering:
    x_ = layers.Permute((2, 1))(x)
    x_ = layers.Reshape((x_.shape[1], x_.shape[2], 1))(x_)
    x_ = layers.Conv2D(x_.shape[1],
                       kernel_size=(x_.shape[1], 1),
                       strides=(1, 1),
                       kernel_initializer='he_normal')(x_)
    d['conv0'] = layers.Reshape((x_.shape[1]* x_.shape[2], x_.shape[3]))(x_)
    layer_counter = 1

    # ENCODER
    for n in range(num_layers_encoder):
        x = layers.MaxPooling1D(pool_size=2)(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            x = conv_layer(y=x,
                           dilation_rate=dilation_rate,
                           num_channels=init_filter_num + filter_num_iterations * layer_counter,
                           kernel_size=kernel_size)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1

    # DECODER
    for n in range(num_layers_decoder):
        x = layers.UpSampling1D(size=2)(d['conv{0}'.format(layer_counter - 1)])
        x = conv_layer(y=x, num_channels=init_filter_num * (num_layers_encoder - n), kernel_size=2)
        #x = add_common_layers(x)
        x = layers.concatenate([d['conv{0}'.format(num_layers_encoder - (n + 1))], x], axis=2)
        for m in range(layer_depth):
            x = conv_layer(y=x,
                           dilation_rate=1,
                           num_channels=init_filter_num + filter_num_iterations * (num_layers_encoder - (n + 1)),
                           kernel_size=kernel_size)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1
    return x




def unet(x, init_filter_num=16, filter_num_iterations=16, kernel_size=3, cardinality=1, dropout_rate=0.5, layer_depth=2, num_layers_encoder=7,
         num_layers_decoder=2, dilation_rate=1, BN_momentum=0.95, input_dropout=False):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout2D(dropout_rate)(y)
        return y

    def conv_layer(y, num_channels, kernel_size=(1, 3), strides=1, dilation_rate=1, data_format='channels_last'):

        if cardinality == 1:
            return layers.Conv2D(num_channels,
                                 kernel_size=(1, 16),
                                 strides=(1, 1),
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer='he_normal')(y)

        assert not num_channels % cardinality
        _d_out = num_channels // cardinality
        _d_in = y.shape[-1] // cardinality
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, j * _d_in:j * _d_in + _d_in])(y)
            groups.append(layers.Conv1D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y


    d = {}
    # spatial filtering:
    x_ = layers.Permute((2, 1))(x)
    x_ = layers.Reshape((x_.shape[1], x_.shape[2], 1))(x_)
    x_ = layers.Conv2D(x_.shape[1],
                       kernel_size=(x_.shape[1], 1),
                       strides=(1, 1),
                       kernel_initializer='he_normal')(x_)
    d['conv0'] = layers.Permute((3, 2, 1))(x_)
    layer_counter = 1

    # ENCODER
    for n in range(num_layers_encoder):
        x = layers.MaxPooling2D(pool_size=(1, 2))(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            x = conv_layer(y=x,
                           dilation_rate=dilation_rate,
                           kernel_size=(1, 16),
                           num_channels=init_filter_num + filter_num_iterations * layer_counter)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1

    # DECODER
    for n in range(num_layers_decoder):
        x = layers.UpSampling2D(size=(1, 2))(d['conv{0}'.format(layer_counter - 1)])
        x = conv_layer(y=x, num_channels=init_filter_num * (num_layers_encoder - n), kernel_size=(1, 2))
        #x = add_common_layers(x)
        x = layers.concatenate([d['conv{0}'.format(num_layers_encoder - (n + 1))], x], axis=3)
        for m in range(layer_depth):
            x = conv_layer(y=x,
                           dilation_rate=1,
                           num_channels=init_filter_num + filter_num_iterations * (num_layers_encoder - (n + 1)),
                           kernel_size=kernel_size)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    return x