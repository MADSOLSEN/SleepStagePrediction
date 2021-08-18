from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K
from models import blocks

def unet_model(input_shape,
               number_of_classes,
               init_filter_num,
               filter_increase_factor,
               cardinality,
               dropout_rate,
               kernel_size,
               layer_depth,
               num_layers,
               num_outputs,
               last_dropout_rate=0,
               dilation_rate=1,
               BN_momentum=0.99,
               add_classifier=False,
               output_layer='sigmoid',
               input_dropout=False,
               block_type='conv_1D'):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    # model
    x = unet(x,
             init_filter_num=init_filter_num,
             filter_increase_factor=filter_increase_factor,
             kernel_size=kernel_size,
             cardinality=cardinality,
             dropout_rate=dropout_rate,
             layer_depth=layer_depth,
             dilation_rate=dilation_rate,
             BN_momentum=BN_momentum,
             num_layers=num_layers,
             input_dropout=input_dropout,
             block_type=block_type)

    if add_classifier:
        x = layers.SpatialDropout1D(last_dropout_rate)(x)
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)
    model = Model(inputs=x_input, outputs=x)
    return model

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


def unet(x, block_type, init_filter_num=8, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
         layer_depth=1, num_layers=10, dilation_rate=1, BN_momentum=0.95, input_dropout=False):

    d = {}
    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)

    if num_layers > 0:
        x = layers.Conv1D(init_filter_num,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(x)

        d['conv0'] = blocks['add_common_layers_1D'](x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
    layer_counter = 1

    # ENCODER
    for n in range(num_layers):
        kernel_size_ = min(x.shape[1], kernel_size)
        x = layers.MaxPooling1D(pool_size=2)(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            x = blocks[block_type](x, )
            if resnet:
                project_shortcut = True if m == 0 else False
                x = residual_block(y=x,
                                   nb_channels_in=x.shape[-1],
                                   nb_channels_out=init_filter_num * filter_increase_factor ** layer_counter,
                                   kernel_size=kernel_size_,
                                   _strides=1,
                                   _project_shortcut=project_shortcut,
                                   bottle_neck=bottle_neck)
            elif wavenet:
                wavenet_residual_block(l_input=x,
                                       num_filters=init_filter_num * filter_increase_factor ** layer_counter,
                                       kernel_size=kernel_size_,
                                       dilation_rate=dilation_rate)
            else:
                x = conv_layer(y=x,
                               dilation_rate=dilation_rate,
                               num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                               kernel_size=kernel_size_)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1


    return x
