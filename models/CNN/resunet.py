from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import StochasticDepth
import tensorflow.keras.backend as K

def ResUNet(input_shape,
              number_of_classes,
              num_outputs,
              output_layer='sigmoid',
              weight_decay=0.00,
              momentum=0.99,
              init_filter_num=8,
              filter_increment_factor=2 ** (1 / 2),
              kernel_size=(9, 1),
              max_pool_size=(2, 1),
              max_pool_increment_iteration=20,
              auxiliary_concatenate=False,
              auxiliary_aux_downsample=False,
              auxiliary_input_downsample=False,
              aux_softmax=False,
              residual=True,
              residual_connection=False,
              activation='gelu',
              attention=False,
              final_cnn=True,
              final_split=False,
              data_format='channels_last',
              maxpool=False,
              dilation_rates=[],
              ConvNeXt_residual=False,
              inverted_bottleneck=False,
              stochastic_depth=True,
              deconvolution=True,
              depth=None,
              normalization='batchnorm'):

    # OPO
    #init_filter_num = 8
    #filter_increment_factor = 2 ** (1 / 3)
    #max_pool_size = (2, 1)
    #depth = 15
    #kernel_size = (16, 1)
    k = 1
    if depth is None:
        depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0], max_pool_increment_iteration=max_pool_increment_iteration)

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    norm = layers.BatchNormalization
    if normalization == 'layernorm':
        norm = layers.LayerNormalization

    # zeropad:
    zeros_to_add = int(2 ** (np.ceil(np.log2(input_shape[0]))) - input_shape[0])
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.ZeroPadding2D(padding=(zeros_to_add // 2, 0))(x)

    if auxiliary_aux_downsample:
        x_aux_input = layers.Input(shape=(num_outputs // 2, 4))
        x_aux = x_aux_input
        if aux_softmax:
            x_aux = layers.Softmax()(x_aux)
        x_aux = layers.UpSampling1D(size=2)(x_aux)

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
        x = conv_block(x=x, features=int(features), kernel_size=kernel_size, normalization=normalization, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)
        skips.append(x)
        features *= filter_increment_factor

        # subsample:
        if maxpool:
            # TODO - testing
            x = layers.MaxPool2D(max_pool_size, data_format=data_format)(x)
        else:
            x = layers.Conv2D(int(features),
                              kernel_size=max_pool_size,
                              activation=None,
                              padding='same',
                              strides=max_pool_size,
                              data_format=data_format,
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer=l2(weight_decay))(x)

        x = norm()(x)

        # update parameters
        if (i + 1) % max_pool_increment_iteration == 0:
            max_pool_size = (max_pool_size[0] * 2, max_pool_size[1])
        if x.shape[2] / max_pool_size[1] < 1:
            max_pool_size = (max_pool_size[0], 1)
        kernel_size = [min(ks, x_dim) for ks, x_dim in zip(kernel_size, x.shape[1:3])]

    # Middel part
    # ========================================================
    x = conv_block(x=x, features=int(features), kernel_size=kernel_size, normalization=normalization, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual)

    # Decoder
    # ========================================================
    for count, i in enumerate(reversed(range(depth))):

        # upsample and convolve
        if attention:
            x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size_list[i], data_format=data_format)
        else:
            x = layers.Conv2DTranspose(features_list[i],
                                       kernel_size=[int(mp) for mp in max_pool_size_list[i]],
                                       strides=[int(mp) for mp in max_pool_size_list[i]],
                                       padding='same',
                                       activation=None,
                                       data_format=data_format)(x)
            x = norm()(x)
            x = layers.concatenate([skips[i], x], axis=3)

        # MobileNetBlock
        x = conv_block(x=x, features=features_list[i], kernel_size=kernel_size_list[i], normalization=normalization, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)

    # Format the segment:
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.Lambda(lambda z: z[:, zeros_to_add // 2: - zeros_to_add // 2, :, :])(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    # final nonlinear stuff:
    if final_cnn:
        x = layers.Conv1D(filters=init_filter_num ,
                          kernel_size=1,
                          padding='same',
                          activation=activation,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay)
                          )(x)

    if input_shape[0] // num_outputs > 0:
        x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)

    if auxiliary_concatenate:
        x_aux_input = layers.Input(shape=(num_outputs // 2, 4))
        x_aux = x_aux_input
        upsample_factor = x.shape[1] // x_aux.shape[1]
        x_aux = layers.UpSampling1D(size=round(upsample_factor))(x_aux)
        x = layers.Concatenate(axis=-1)([x, x_aux])

    if auxiliary_aux_downsample and x.shape[1] == num_outputs:
        x_con = []
        for n in range(x_aux.shape[-1]):
            x_con += [layers.Multiply()([x, x_aux[:, :, n:n+1]])]
        x = layers.Concatenate(axis=-1)(x_con + [x])

    if final_cnn:
        x = layers.Conv1D(filters=number_of_classes,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)

    # Classifier
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer)(x)

    if any([auxiliary_concatenate, auxiliary_aux_downsample, auxiliary_input_downsample]):
        model = Model(inputs=[x_input, x_aux_input], outputs=x)
    else:
        model = Model(inputs=x_input, outputs=x)

    return model


def conv_block(x, features, kernel_size, normalization='batchnorm', data_format='channels_last', weight_decay=0.,
               residual=True, stochastic_depth=True, activation='gelu'):

    if normalization == 'batchnorm':
        norm = layers.BatchNormalization
    if normalization == 'layernorm':
        norm = layers.LayerNormalization

    x_ = layers.Conv2D(int(features),
                       kernel_size=kernel_size,
                       activation=activation,
                       padding='same',
                       data_format=data_format,
                       kernel_regularizer=l2(weight_decay),
                       bias_regularizer=l2(weight_decay))(x)
    if normalization is not None:
        x_ = norm()(x_)

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

def determine_depth(temporal_shape, temporal_max_pool_size, max_pool_increment_iteration):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= round(temporal_max_pool_size)
        if depth % max_pool_increment_iteration == 0:
            temporal_max_pool_size *= temporal_max_pool_size
    depth -= 1
    return depth

