from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def att_r2_unet(input_shape,
                number_of_classes,
                num_outputs,
                depth,
                weight_regularization=0,
                init_filter_num=32,
                max_pool_size=2,
                dropout=0.0,
                batch_normalization=False,
                layer_normalization=False,
                output_layer='sigmoid',
                data_format='channels_last'):
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    # x = layers.Reshape((x_input.shape[1], x_input.shape[2], 1))(x)  # to 2D:
    x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)  # to 2D:

    features = init_filter_num
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization)
        x = layers.Dropout(dropout)(x)
        skips.append(x)
        x = layers.MaxPooling2D((max_pool_size, 1), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization)
    x = layers.Dropout(dropout)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size, data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization)
        x = layers.Dropout(dropout)(x)

    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)  # to 1D:

    x = layers.Dense(units=number_of_classes,
                     activation='sigmoid',
                     kernel_initializer='he_normal')(x)
    x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)

    return Model(inputs=x_input, outputs=x)