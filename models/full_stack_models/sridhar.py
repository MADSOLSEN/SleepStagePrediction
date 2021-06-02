from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..blocks import add_common_layers_2D, conv_chunks_layer_2D
import numpy as np
from tensorflow import extract_volume_patches
import tensorflow as tf
import tensorflow.keras.backend as K


def sridhar_model(input_shape,
                  number_of_classes,
                  num_outputs,
                  init_filter_num=8,
                  kernel_size=3,
                  num_layers=3,
                  kernel_size_decrease_factor=1,
                  filter_increase_factor=1,
                  dropout_rate=0,
                  layer_depth=1,
                  cardinality=1,
                  bottle_neck=False,
                  output_layer='sigmoid',
                  dilation_rate=1,
                  BN_momentum=0.99,
                  input_dropout=False,
                  block_type='conv_1D',
                  weight_regularization=0,
                  block_size=30*4,
                  block_stride=30,
                  fs=4):
    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)

    # adjust time axis:
    adjust_size = False
    adjusted_temporal_input_size = input_shape[0]
    if adjust_size:
        adjusted_temporal_input_size = int(2 ** np.ceil(np.log2(input_shape[0])))
        padding = (adjusted_temporal_input_size - input_shape[0]) // 2
        x = layers.ZeroPadding1D(padding=padding)(x)

    x = overlapping_blocker(x, block_size=block_size * fs, stride=block_stride * fs)

    # input kernel
    x = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same')(x)

    # Local convolutional part
    for layer_num in range(2):
        skip = layers.MaxPool1D(pool_size=2, strides=2)(x)
        for depth_num in range(2):
            x = layers.Conv1D(filters=16, kernel_size=6, strides=1, padding='same')(x)
            x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.MaxPool1D(pool_size=2, strides=2)(x)
        x = layers.Add()([x, skip])

    x = tf.reshape(x, [-1, num_outputs, x.shape[1] * x.shape[2]])
    x = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(x)

    # Temporal part
    for layer_num in range(2):
        x_list = [layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(x)]
        for dr in [2, 4, 8, 16, 32]:
            x_ = layers.Conv1D(filters=64, kernel_size=14, dilation_rate=dr, padding='same')(x)
            x_ = layers.LeakyReLU(alpha=0.15)(x_)
            x_list += [layers.Dropout(rate=0.2)(x_)]
        x = layers.Add()(x_list)

    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)

    model = Model(inputs=x_input, outputs=x)
    return model

def overlapping_blocker(tensor, block_size, stride):

    test = False
    if test:

        block_size_ = 8
        stride_ = 4
        num_outputs = 8

        # test with knows sizes:
        input1 = tf.constant(list(range(32 * 1)))
        input2 = tf.constant(list(range(32 * 1)))
        input1 = tf.reshape(input1, [1, 32, 1])
        input2 = tf.reshape(input2, [1, 32, 1])
        input = tf.concat([input1, input2], axis=0)
        x = tf.extract_volume_patches(input[..., None, None],
                                      ksizes=[1, block_size_, 1, 1, 1],
                                      strides=[1, stride_, 1, 1, 1],
                                      padding='SAME')
        x_reshaped = tf.reshape(x, [-1, block_size_, input.shape[-1]])

        # model
        y = layers.MaxPooling1D(pool_size=2)(x_reshaped)

        # reshape output:
        # x_temp0 = tf.reshape(y, [-1, y.shape[1] * y.shape[2]])
        x_out = tf.reshape(x, [-1, num_outputs, y.shape[1] * y.shape[2]])
        k = 1

    k = 1
    x = tf.extract_volume_patches(tensor[..., None, None],
                                  ksizes=[1, block_size, 1, 1, 1],
                                  strides=[1, stride, 1, 1, 1],
                                  padding='SAME')
    x_reshaped = tf.reshape(x, [-1, tensor.shape[-1], block_size])
    x_perm = tf.transpose(x_reshaped, perm=[0, 2, 1])

    # x_reshaped = tf.reshape(x, [-1, block_size, tensor.shape[-1]])
    return x_perm


def zeropad(x, zeros_to_add):
    x = layers.ZeroPadding1D(padding=zeros_to_add)(x)
    return x



def cnn1d(x, block_type, init_filter_num=8, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
          layer_depth=1, num_layers=10, dilation_rate=1, BN_momentum=0.95, input_dropout=False, bottle_neck=False,
          weight_regularization=0, kernel_size_decrease_factor=1):

    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)

    layer_counter = 0
    for n in range(num_layers):
        kernel_size_ = min(x.shape[1], kernel_size)

        if block_type == 'resnet':
            for m in range(layer_depth):
                if m == 0: y = x
                y = conv_chunks_layer_2D(y=y, # TODO solve x!
                                         dilation_rate=dilation_rate,
                                         num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                         kernel_size=(1, kernel_size_),
                                         weight_regularization=weight_regularization)
                y = add_common_layers_2D(y, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
            y = layers.MaxPooling2D(pool_size=(1, 2))(y)
            x = conv_chunks_layer_2D(x, num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                     kernel_size=1, strides=(1, 2), dilation_rate=1, cardinality=cardinality,
                                     weight_regularization=weight_regularization)
            x = layers.add([x, y])
        else:
            for m in range(layer_depth):
                x = conv_chunks_layer_2D(y=x,
                                         dilation_rate=dilation_rate,
                                         num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                         kernel_size=(1, kernel_size_),
                                         weight_regularization=weight_regularization)
                x = add_common_layers_2D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
            x = layers.MaxPooling2D(pool_size=(1, 2))(x)
            kernel_size = max(3, kernel_size // kernel_size_decrease_factor)
        layer_counter += 1
    return x
