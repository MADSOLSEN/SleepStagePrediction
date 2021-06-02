from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

# downloaded from:
# https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
# keras implementation of U-Net R2U attention.
# Based on the original paper: "Attention U-Net: where to look for the Pancreas".


def up_and_concate(down_layer, layer, upsample_size=2, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]


    up = layers.UpSampling2D(size=(upsample_size, 1), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer, upsample_size=2, data_format='channels_last', activation='relu'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = layers.UpSampling2D(size=upsample_size, data_format=data_format)(down_layer)
    up = layers.Conv2D(down_layer.shape[-1], upsample_size, activation=activation, padding='same', data_format=data_format)(up)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = layers.Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = layers.Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = layers.multiply([x, rate])

    return att_x


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 1], stride=[1, 1],
              padding='same', data_format='channels_last', activation='relu'):

    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = layers.Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = layers.BatchNormalization()(layer)
        layer = layers.Activation(activation)(layer)
        layer = layers.Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = layers.Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = layers.Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = layers.add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 1], stride=[1, 1],
                  padding='same', data_format='channels_last', layer_normalization=False, activation='relu',
                  weight_regularization=1e-5):

    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = layers.Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = layers.Conv2D(out_n_filters,
                                       kernel_size,
                                       strides=stride,
                                       padding=padding,
                                       data_format=data_format,
                                       kernel_regularizer=l2(weight_regularization),
                                       bias_regularizer=l2(weight_regularization))(layer)
                if batch_normalization:
                    layer1 = layers.BatchNormalization()(layer1)
                if layer_normalization:
                    layer1 = layers.LayerNormalization()(layer1)
                layer1 = layers.Activation(activation)(layer1)
            layer1 = layers.Conv2D(out_n_filters,
                                   kernel_size,
                                   strides=stride,
                                   padding=padding,
                                   data_format=data_format,
                                   kernel_regularizer=l2(weight_regularization),
                                   bias_regularizer=l2(weight_regularization))(layers.add([layer1, layer]))
            if batch_normalization:
                layer1 = layers.BatchNormalization()(layer1)
            if layer_normalization:
                layer1 = layers.LayerNormalization()(layer1)
            layer1 = layers.Activation(activation)(layer1)
        layer = layer1

    out_layer = layers.add([layer, skip_layer])
    return out_layer