from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def att_unet(input_shape,
             number_of_classes,
             num_outputs,
             depth,
             weight_regularization=0,
             init_filter_num=32,
             max_pool_size=2,
             dropout=0.0,
             output_layer='sigmoid',
             data_format='channels_last'):

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)  # to 2D:

    features = init_filter_num
    skips = []
    for i in range(depth):
        x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = layers.MaxPooling2D((max_pool_size, 1), data_format=data_format)(x)
        features = features * 2

    x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size, data_format=data_format)
        x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv2D(features, (3, 1), activation='relu', padding='same', data_format=data_format)(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)  # to 1D:
    x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    return Model(inputs=x_input, outputs=x)