from tensorflow.keras import layers
from tensorflow.keras.models import Model
from ..blocks_downloaded import CausalAtrousConvolution1D
from keras.regularizers import l2
import numpy as np
from tensorflow.keras.regularizers import l2


def wavenet_model(input_shape,
                  number_of_classes,
                  num_outputs,
                  init_filter_num=64,
                  nb_output_bins=256,
                  dilation_depth=9,
                  use_skip_connections=True,
                  layer_depth=1,
                  last_dropout_rate=0,
                  output_layer='sigmoid',
                  weight_regularization=0,
                  input_dropout_rate=0,
                  output_dropout_rate=0
                  ):
    # author: https://github.com/basveeling/wavenet/blob/master/wavenet_utils.py
    #
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Dropout(input_dropout_rate)(x)

    x = wavenet(x,
                nb_stacks=layer_depth,
                nb_filters=init_filter_num,
                nb_output_bins=nb_output_bins,
                dilation_depth=dilation_depth,
                use_skip_connections=use_skip_connections,
                use_bias=True,
                res_l2=weight_regularization,
                final_l2=weight_regularization)

    x = layers.Dropout(output_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    return Model(inputs=x_input, outputs=x)

def wavenet(x, nb_stacks=1, nb_filters=32, nb_output_bins=256, dilation_depth=9, use_skip_connections=True,
            use_bias=True, res_l2=0, final_l2=0):

    def residual_block(x):
        original_x = x
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = layers.Conv1D(nb_filters, 2, dilation_rate=2 ** i, padding='same', use_bias=use_bias,
                        name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh', kernel_regularizer=l2(res_l2))(x)
        sigm_out = layers.Conv1D(nb_filters, 2, dilation_rate=2 ** i, padding='same', use_bias=use_bias,
                        name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid', kernel_regularizer=l2(res_l2))(x)
        x = layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])
        res_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                     kernel_regularizer=l2(res_l2))(x)
        skip_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                      kernel_regularizer=l2(res_l2))(x)
        res_x = layers.Add()([original_x, res_x])
        return res_x, skip_x


    skip_connections = []
    x = layers.Conv1D(nb_filters, 2, dilation_rate=1, padding='same', name='initial_causal_conv')(x)
    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(x)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = layers.Add()(skip_connections)
    x = layers.Activation('relu')(x)
    x = layers.Convolution1D(nb_output_bins, 1, padding='same',
                               kernel_regularizer=l2(final_l2))(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution1D(nb_output_bins, 1, padding='same')(x)

    return x


def wavenet_ori(x, nb_stacks=1, nb_filters=32, nb_output_bins=256, dilation_depth=9, use_skip_connections=True,
            use_bias=True, res_l2=0, final_l2=0):

    def residual_block(x):
        original_x = x
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalAtrousConvolution1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True, use_bias=use_bias,
                        name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh', kernel_regularizer=l2(res_l2))(x)
        sigm_out = CausalAtrousConvolution1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True, use_bias=use_bias,
                        name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid', kernel_regularizer=l2(res_l2))(x)
        x = layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])
        res_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                     kernel_regularizer=l2(res_l2))(x)
        skip_x = layers.Convolution1D(nb_filters, 1, padding='same', use_bias=use_bias,
                                      kernel_regularizer=l2(res_l2))(x)
        res_x = layers.Add()([original_x, res_x])
        return res_x, skip_x


    skip_connections = []
    x = CausalAtrousConvolution1D(nb_filters, 2, dilation_rate=1,
             padding='valid', causal=True, name='initial_causal_conv')(x)
    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(x)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = layers.Add()(skip_connections)
    x = layers.Activation('relu')(x)
    x = layers.Convolution1D(nb_output_bins, 1, padding='same',
                               kernel_regularizer=l2(final_l2))(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution1D(nb_output_bins, 1, padding='same')(x)

    return x
