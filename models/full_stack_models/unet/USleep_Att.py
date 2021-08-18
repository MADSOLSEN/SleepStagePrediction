from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K

def USleep_Att(input_shape,
               number_of_classes,
               num_outputs,
               output_layer='softmax',
               weight_decay=0.0,
               init_filter_num=32,
               filter_increment_factor=2 ** (1/2),
               kernel_size=(32, 3),
               max_pool_size=(4, 2),
               max_pool_increment_iteration=20,
               dropout=0.0,
               layer_normalization=True,
               activation='elu',
               attention=True,
               final_cnn=True,
               spatial_filtering=False,
               spatial_dropout=False,
               data_format='channels_last',
               dilation_rates=[]):

    depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0], max_pool_increment_iteration=max_pool_increment_iteration)

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    features = init_filter_num
    skips = []
    features_list = []
    kernel_size_list = []
    max_pool_size_list = []

    if spatial_filtering:
        """
        x_ = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                    activation=activation,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=l2(weight_decay),
                                    bias_regularizer=l2(weight_decay))(x)
        if layer_normalization:
            x_ = layers.LayerNormalization()(x_)
        """
        x_ = layers.Add()([x[:, :, :, 0], x[:, :, :, 1]])
        x = layers.Reshape((x.shape[1], x.shape[2], 1))(x_)



    if spatial_dropout:
        x = layers.SpatialDropout2D(rate=0.05)(x)

    # Encoder
    # ========================================================
    for i in range(depth):
        if len(dilation_rates) > 0:
            x = conv_stacked_dilated_2D(x,
                                        num_channels=int(features),
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        data_format=data_format,
                                        weight_decay=weight_decay,
                                        dilation_rates=dilation_rates)
        else:
            x = layers.Conv2D(int(features),
                              kernel_size=kernel_size,
                              activation=activation,
                              padding='same',
                              data_format=data_format,
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer=l2(weight_decay))(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)

        # append lists
        skips.append(x)
        max_pool_size_list.append(max_pool_size)
        kernel_size_list.append(kernel_size)
        features_list.append(features)

        x = layers.MaxPooling2D(max_pool_size, data_format=data_format)(x)

        # update parameters
        if (i + 1) % max_pool_increment_iteration == 0:
            max_pool_size = (max_pool_size[0] * 2, max_pool_size[1])
        if x.shape[2] / max_pool_size[1] < 1:
            max_pool_size = (max_pool_size[0], 1)
        kernel_size = [min(ks, x_dim) for ks, x_dim in zip(kernel_size, x.shape[1:3])]

        # kernel_size[0] = min(kernel_size[0], x.shape[1])
        features *= filter_increment_factor

    if len(dilation_rates) > 0:
        x = conv_stacked_dilated_2D(x,
                                    num_channels=int(features),
                                    kernel_size=kernel_size,
                                    activation=activation,
                                    data_format=data_format,
                                    weight_decay=weight_decay,
                                    dilation_rates=dilation_rates)
    else:
        x = layers.Conv2D(int(features),
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          kernel_initializer='he_normal',
                          data_format=data_format,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(dropout)(x)
    if layer_normalization:
        x = layers.LayerNormalization()(x)

    # max_pool_size_list.append([1, 1]) # TODO - update this to
    #features_list.append(features)

    # decoder
    for i in reversed(range(depth)):

        if attention:
            x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size_list[i], data_format=data_format)
        else:
            x = layers.UpSampling2D(size=[int(mp) for mp in max_pool_size_list[i]], data_format=data_format)(x)
            x = layers.Conv2D(features, kernel_size=[int(mp) for mp in max_pool_size_list[i]], activation=activation, padding='same', data_format=data_format)(x)
            x = layers.concatenate([skips[i], x], axis=3)

        if len(dilation_rates) > 0:
            x = conv_stacked_dilated_2D(x,
                                        num_channels=int(features_list[i]),
                                        kernel_size=(kernel_size_list[i]),
                                        activation=activation,
                                        data_format=data_format,
                                        weight_decay=weight_decay,
                                        dilation_rates=dilation_rates)
        else:
            x = layers.Conv2D(int(features_list[i]),
                              kernel_size=kernel_size_list[i],
                              activation=activation,
                              padding='same',
                              kernel_initializer='he_normal',
                              data_format=data_format,
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer=l2(weight_decay))(x)
        x = layers.Dropout(dropout)(x)
        if layer_normalization:
            x = layers.LayerNormalization()(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    if final_cnn:
        x = layers.Conv1D(filters=init_filter_num,
                          kernel_size=1,
                          padding='same',
                          activation=activation,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay)
                          )(x)

    if input_shape[0] // num_outputs > 0:
        x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)

    if final_cnn:
        x = layers.Conv1D(filters=number_of_classes,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)


    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)


    model = Model(inputs=x_input, outputs=x)
    k = 1
    return model

def determine_depth(temporal_shape, temporal_max_pool_size, max_pool_increment_iteration):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= round(temporal_max_pool_size)
        if depth % max_pool_increment_iteration == 0:
            temporal_max_pool_size *= temporal_max_pool_size
    depth -= 1
    return depth

def conv_stacked_dilated_2D(x, num_channels, kernel_size, activation, data_format, weight_decay, dilation_rates):

    """
    x = layers.Conv2D(filters=num_channels,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=activation,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay)
                          )(x)
    """
    x_out = []
    for dilation_rate in dilation_rates:
        """
        x = layers.Conv2D(filters=num_channels // 4,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=activation,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay)
                          )(x)
        """
        x = layers.Conv2D(num_channels,
                           kernel_size=kernel_size,
                           activation=activation,
                           dilation_rate=dilation_rate,
                           padding='same',
                           data_format=data_format,
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay))(x)
        #x_out.append(x_)
    #x_out = layers.Add()(x_out)
    #x_out = layers.concatenate(x_out)
    x_out = x
    return x_out
