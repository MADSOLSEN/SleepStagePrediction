from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from ..blocks import add_common_layers_1D, conv_layer_1D, residual_block_1D, wavenet_residual_block_1D
import tensorflow as tf
import tensorflow.keras.backend as K

def cnn1d_spatial_model(input_shape,
                init_filter_num,
                kernel_size,
                num_layers,
                num_outputs,
                filter_increase_factor=1,
                dropout_rate=0,
                layer_depth=1,
                cardinality=1,
                bottle_neck=False,
                dilation_rate=1,
                BN_momentum=0.95,
                input_dropout=False,
                block_type='conv_1D',
                weight_regularization=0):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    x = cnn1d(x,
              init_filter_num=init_filter_num,
              filter_increase_factor=filter_increase_factor,
              kernel_size=kernel_size,
              cardinality=cardinality,
              bottle_neck=bottle_neck,
              dropout_rate=dropout_rate,
              layer_depth=layer_depth,
              dilation_rate=dilation_rate,
              BN_momentum=BN_momentum,
              num_layers=num_layers,
              input_dropout=input_dropout,
              block_type=block_type,
              weight_regularization=weight_regularization)

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


def cnn1d(x, block_type, init_filter_num=8, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
          layer_depth=1, num_layers=10, dilation_rate=1, BN_momentum=0.95, input_dropout=False, bottle_neck=False,
          weight_regularization=0):

    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)

    # spatial filtering:
    x_ = layers.Permute((2, 1))(x)
    x_ = layers.Reshape((x_.shape[1], x_.shape[2], 1))(x_)
    x_ = layers.Conv2D(x_.shape[1],
                       kernel_size=(x_.shape[1], 1),
                       strides=(1, 1),
                       kernel_initializer='he_normal')(x_)
    x_ = layers.Permute((2, 3, 1))(x_)
    x = layers.Reshape((K.int_shape(x_)[1], K.int_shape(x_)[2] * K.int_shape(x_)[3]))(x_)
    layer_counter = 1


    layer_counter = 0
    for n in range(num_layers):
        kernel_size_ = min(x.shape[1], kernel_size)
        for m in range(layer_depth):
            if block_type=='resnet':
                project_shortcut = True if m == 0 else False
                x = residual_block_1D(y=x,
                                      num_channels_in=x.shape[-1],
                                      num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                      kernel_size=kernel_size_,
                                      strides=1,
                                      _project_shortcut=project_shortcut,
                                      bottle_neck=bottle_neck,
                                      cardinality=cardinality,
                                      weight_regularization=weight_regularization)
            elif block_type == 'wavenet':
                wavenet_residual_block_1D(y=x,
                                          num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                          kernel_size=kernel_size_,
                                          dilation_rate=dilation_rate,
                                          weight_regularization=weight_regularization)
            else:
                x = conv_layer_1D(y=x,
                                  dilation_rate=dilation_rate,
                                  num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                  kernel_size=kernel_size_,
                                  weight_regularization=weight_regularization)
            x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
        x = layers.MaxPooling1D(pool_size=2)(x)
        layer_counter += 1
    return x
