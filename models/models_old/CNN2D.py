from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def cnn2D_model(input_shape,
                batch_size,
                number_of_classes,
                init_filter_num,
                filter_increase_factor,
                cardinality,
                dropout_rate,
                kernel_size,
                layer_depth,
                num_layers,
                num_outputs,
                dilation_rate=1,
                input_dropout=False,
                BN_momentum=0.97,
                add_Dense=False,
                add_rnn=False,
                add_classifier=False,
                output_layer='sigmoid',
                resnet=False,
                bottle_neck=False,
                wavenet=False,):
    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input
    if len(input_shape) == 2:
        x = layers.Reshape(target_shape=(input_shape + [1]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    # model
    x = cnn2D(x,
              init_filter_num=init_filter_num,
              batch_size=batch_size,
              filter_increase_factor=filter_increase_factor,
              kernel_size=kernel_size,
              cardinality=cardinality,
              dropout_rate=dropout_rate,
              layer_depth=layer_depth,
              num_layers=num_layers,
              BN_momentum=BN_momentum,
              dilation_rate=dilation_rate,
              resnet = resnet,
              bottle_neck = bottle_neck,
              wavenet=wavenet)

    x = layers.Reshape((K.int_shape(x)[1], K.int_shape(x)[2] * K.int_shape(x)[3]))(x)

    if add_rnn:
        x = RNN(x,
                init_filter_num=64,
                dropout_rate=dropout_rate,
                layer_depth=1,
                BN_momentum=BN_momentum)

    if add_classifier:

        num_class_adj = np.max((3, number_of_classes))
        pi = 1 / (num_class_adj)
        b = -np.log((1 - pi) / pi)

        if add_Dense:
            x = dense_layer(x,
                            width=512,
                            dropout_rate=dropout_rate,
                            layer_depth=layer_depth,
                            BN_momentum=BN_momentum)

        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)

    l = 1
    return Model(inputs=x_input, outputs=x)

def zeropad(x, num_outputs, num_halfs):
    adjusted_input_size = num_outputs * 2 ** num_halfs
    assert (adjusted_input_size >= x.shape[1])
    if adjusted_input_size > x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding2D(padding=(zeros_to_add, 0))(x) # only padding the time axis!
    #if x.shape[2] < 2 ** num_halfs:
    #    zeros_to_add = (2 ** num_halfs - x.shape[2]) // 2
    #    x = layers.ZeroPadding2D(padding=(0, zeros_to_add))(x)  # only padding the time axis!
    #k = 1
    return x

def RNN(x, init_filter_num=64, dropout_rate=0.5, layer_depth=1, BN_momentum=0.99):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(0)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Bidirectional(layers.GRU(init_filter_num,
                                            return_sequences=True,
                                            kernel_initializer="he_normal"))(x)

        x = add_common_layers(x)

    return x

def dense_layer(x, width, dropout_rate, layer_depth, BN_momentum=0.99):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        # y = layers.LeakyReLU()(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Dense(units=width,
                         #padding='same',
                         kernel_initializer='he_normal')(x)
        x = add_common_layers(x)
    return x

def cnn2D(x, init_filter_num=16, batch_size=32, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0.5,
          layer_depth=2, num_layers=5, dilation_rate=(1, 1), BN_momentum=0.95, input_dropout=False, resnet=False,
          bottle_neck=False, wavenet=False):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        # y = layers.LeakyReLU()(y)
        y = swish(y)
        y = layers.SpatialDropout2D(dropout_rate)(y)
        return y

    def conv_layer(y, num_channels, kernel_size, strides=(1, 1), dilation_rate=(1, 1)):

        if cardinality == 1:
            return layers.Conv2D(num_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer='he_normal')(y)

        assert not num_channels % cardinality
        _d_out = num_channels // cardinality
        _d_in = y.shape[-1] // cardinality
        # _b_size = batch_size // cardinality
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d_in:(j + 1) * _d_in])(y)
            groups.append(layers.Conv2D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, kernel_size, _strides=(1, 1), _project_shortcut=False, bottle_neck=False):

        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """

        shortcut = y

        if bottle_neck:
            # we modify the residual building block as a bottleneck design to make the network more economical
            y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            y = add_common_layers(y)

            # ResNeXt (identical to ResNet when `cardinality` == 1)
            y = conv_layer(y, num_channels=nb_channels_in, kernel_size=kernel_size, strides=1)
            y = add_common_layers(y)

            y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = add_common_layers(y)
        else:
            y = conv_layer(y, num_channels=nb_channels_out, kernel_size=kernel_size, strides=1, dilation_rate=(1, 1))
            y = add_common_layers(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut:
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = add_common_layers(shortcut)

        y = layers.add([shortcut, y])
        return y

    def wavenet_residual_block(l_input, num_filters, kernel_size, dilation_rate):
        # Gated activation.
        l_sigmoid_conv1d = layers.Conv2D(num_filters,
                                         kernel_size=kernel_size,
                                         dilation_rate=dilation_rate,
                                         padding="same",
                                         activation="sigmoid")(l_input)
        l_tanh_conv1d = layers.Conv2D(num_filters,
                                      kernel_size=kernel_size,
                                      dilation_rate=dilation_rate,
                                      padding="same",
                                      activation="tanh")(l_input)
        l_mul = layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
        # Branches out to skip unit and residual output.
        l_skip_connection = layers.Conv2D(1, (1, 1))(l_mul)
        l_residual = layers.Add()([l_input, l_skip_connection])
        return l_residual, l_skip_connection


    d = {}
    x = layers.Conv2D(init_filter_num,
                      kernel_size=(kernel_size, 3),
                      strides=(1, 1),
                      padding='same',
                      dilation_rate=dilation_rate,
                      kernel_initializer='he_normal')(x)
    d['conv0'] = add_common_layers(x)
    layer_counter = 1

    # ENCODER
    for n in range(num_layers):
        temporal_kernel_size = min(x.shape[1], kernel_size)
        kernel_size_ = (temporal_kernel_size, 1) if x.shape[-2] <= 1 else (temporal_kernel_size, 3)
        pool_size = (2, 1) if x.shape[-2] <= 1 else (2, 2)

        x = layers.MaxPooling2D(pool_size=pool_size)(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            if resnet:
                project_shortcut = True if m == 0 else False
                x = residual_block(y=x,
                                   nb_channels_in=x.shape[-1],
                                   nb_channels_out=init_filter_num * filter_increase_factor ** layer_counter,
                                   kernel_size=kernel_size_,
                                   _strides=(1, 1),
                                   _project_shortcut=project_shortcut,
                                   bottle_neck=bottle_neck)
            elif wavenet:
                wavenet_residual_block(l_input=x,
                                       num_filters=init_filter_num * filter_increase_factor ** layer_counter,
                                       kernel_size=kernel_size_,
                                       dilation_rate=dilation_rate)
            else:
                x = conv_layer(y=x,
                               dilation_rate=dilation_rate,
                               num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                               kernel_size=kernel_size_)
                x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1
    return x