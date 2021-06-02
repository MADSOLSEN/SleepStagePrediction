from .lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def att_r2_unet(input_shape,
                number_of_classes,
                num_outputs,
                depth,
                weight_regularization=0,
                init_filter_num=32,
                filter_increment_factor=2,
                kernel_size=(3, 1),
                max_pool_size=2,
                dropout=0.0,
                batch_normalization=False,
                layer_normalization=True,
                activation='elu',
                output_layer='sigmoid',
                softmax=True,
                final_cnn=False,
                class_attention=False,
                multiheadattention=False,
                data_format='channels_last'):
    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input
    x = layers.Reshape((x_input.shape[1], 1, x_input.shape[2]))(x)

    #features_list = [init_filter_num]
    features = init_filter_num

    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, kernel_size=kernel_size, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization, activation=activation)
        x = layers.Dropout(dropout)(x)
        skips.append(x)
        x = layers.MaxPooling2D((max_pool_size, 1), data_format=data_format)(x)

        features = features * filter_increment_factor

    x = rec_res_block(x, features, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization, activation=activation)
    x = layers.Dropout(dropout)(x)

    for i in reversed(range(depth)):
        features = features / 2
        x = attention_up_and_concate(x, skips[i], upsample_size=max_pool_size, data_format=data_format, activation=activation)
        x = rec_res_block(x, round(features), kernel_size=kernel_size, data_format=data_format, batch_normalization=batch_normalization, layer_normalization=layer_normalization, activation=activation)
        x = layers.Dropout(dropout)(x)

    # Reshape to 1D:
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    if final_cnn:
        x = layers.Conv1D(number_of_classes,
                          kernel_size=1,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation=activation)(x)

    if softmax:
        x_sigmoid = layers.Dense(units=number_of_classes-4,
                         activation='sigmoid',
                         kernel_initializer='he_normal')(x)
        x_softmax = layers.Dense(units=4,
                         activation='softmax',
                         kernel_initializer='he_normal')(x)
        x = layers.Concatenate(axis=-1)([x_sigmoid, x_softmax])
    else:
        x = layers.Dense(units=number_of_classes,
                         activation='sigmoid',
                         kernel_initializer='he_normal')(x)

    if class_attention:
        # inspiration: https://keras.io/examples/vision/image_classification_with_vision_transformer/

        x_T = layers.Permute((2, 1))(x)
        b = layers.Dot(axes=(2, 1))([x_T, x])  # covariance matrix. Holds how each class are correlated across timesteps
        # Think about scaling this - maybe with softmax to make them attention based!
        x_att = layers.Dot(axes=2)([x, b]) # multiply to get the new thing
        x_att_norm = layers.Dense(number_of_classes,
                             activation='softmax',
                             kernel_initializer='he_normal')(x_att)
        x = layers.Add()([x, x_att_norm])

        # new output
        x_sigmoid = layers.Dense(units=number_of_classes - 4,
                                 activation='sigmoid',
                                 kernel_initializer='he_normal')(x)
        x_softmax = layers.Dense(units=4,
                                 activation='softmax',
                                 kernel_initializer='he_normal')(x)
        x = layers.Concatenate(axis=-1)([x_sigmoid, x_softmax])

    if multiheadattention:
        for n in range(1):
            # layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            # create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(x1, x1) # Attention applied to the feature (class) axis!
            # skip connection 1.
            x2 = layers.Add()([x, attention_output])
            # layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = layers.Dense(6, activation='relu')(x3)
            # skip connection 2.
            x = layers.Add()([x3, x2])



    x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)




    return Model(inputs=x_input, outputs=x)