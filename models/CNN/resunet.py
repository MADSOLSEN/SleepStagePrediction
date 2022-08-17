from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import StochasticDepth

def ResUNet(input_shape,
            num_classes,
            num_outputs,
            depth=None,
            init_filter_num=8,
            filter_increment_factor=2 ** (1 / 3),
            kernel_size=(16, 1),
            max_pool_size=(2, 1),
            activation='gelu',
            output_layer='sigmoid',
            weight_decay=0.0,
            residual=False,
            stochastic_depth=False,
            data_format = 'channels_last'):


    if depth is None:
        depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0])

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    # zero-pad:
    zeros_to_add = int(2 ** (np.ceil(np.log2(input_shape[0]))) - input_shape[0])
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.ZeroPadding2D(padding=(zeros_to_add // 2, 0))(x)

    # preallocation
    features = init_filter_num
    skips = []
    features_list = []
    kernel_size_list = []
    max_pool_size_list = []


    # Encoder
    # ========================================================
    for i in range(depth):

        # append lists of variables:
        features_list.append(features)
        kernel_size_list.append(kernel_size)
        max_pool_size_list.append(max_pool_size)

        # Feature extractor
        x = conv_block(x=x, features=int(features), kernel_size=kernel_size, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)
        skips.append(x)
        features *= filter_increment_factor

        # Reshape output to subsequent layer:
        x = layers.Conv2D(int(features),
                          kernel_size=max_pool_size,
                          activation=None,
                          padding='same',
                          strides=max_pool_size,
                          data_format=data_format,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)

        # update kernel_size and max_pool_size
        kernel_size = [min(ks, x_dim) for ks, x_dim in zip(kernel_size, x.shape[1:3])]
        if x.shape[2] / max_pool_size[1] < 1:
            max_pool_size = (max_pool_size[0], 1)


    # Middel part
    # ========================================================
    x = conv_block(x=x, features=int(features), kernel_size=kernel_size, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual)


    # Decoder
    # ========================================================
    for count, i in enumerate(reversed(range(depth))):

        # upsample and convolve
        x = layers.Conv2DTranspose(features_list[i],
                                   kernel_size=[int(mp) for mp in max_pool_size_list[i]],
                                   strides=[int(mp) for mp in max_pool_size_list[i]],
                                   padding='same',
                                   activation=None,
                                   data_format=data_format)(x)
        x = layers.BatchNormalization()(x)

        # concatenate with layer from encoder with same dimensionality
        x = layers.concatenate([skips[i], x], axis=3)

        # feature extractor
        x = conv_block(x=x, features=features_list[i], kernel_size=kernel_size_list[i], activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)


    # Cut-off zero-padded segment:
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.Lambda(lambda z: z[:, zeros_to_add // 2: - zeros_to_add // 2, :, :])(x)

    # reshape
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    # non-linear activation:
    x = layers.Conv1D(filters=init_filter_num ,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)

    if input_shape[0] // num_outputs > 0:
        x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)


    # non-linear activation:
    x = layers.Conv1D(filters=num_classes,
                  kernel_size=1,
                  padding='same',
                  activation=activation,
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay)
                  )(x)


    # Classification
    # ========================================================
    x = layers.Dense(units=num_classes,
                     activation=output_layer)(x)

    return Model(inputs=x_input, outputs=x)


def conv_block(x, features, kernel_size, data_format='channels_last', weight_decay=0.0,
               residual=True, stochastic_depth=True, activation='gelu'):

    # feature extractor
    x_ = layers.Conv2D(int(features),
                       kernel_size=kernel_size,
                       activation=activation,
                       padding='same',
                       data_format=data_format,
                       kernel_regularizer=l2(weight_decay),
                       bias_regularizer=l2(weight_decay))(x)
    x_ = layers.BatchNormalization()(x_)

    if residual:
        if x.shape[-1] != x_.shape[-1]:
            x = layers.Conv2D(int(features),
                              kernel_size=(1, 1),
                              activation=None,
                              padding='same',
                              data_format=data_format,
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer=l2(weight_decay))(x)
        if stochastic_depth:
            return StochasticDepth(survival_probability=0.9)([x, x_])
        else:
            return layers.Add()([x, x_])
    else:
        return x_


def determine_depth(temporal_shape, temporal_max_pool_size):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= round(temporal_max_pool_size)
    depth -= 1
    return depth

