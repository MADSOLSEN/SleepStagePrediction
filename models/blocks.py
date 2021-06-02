from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K



def swish(x, beta=1):
    return (x * K.sigmoid(beta * x))


def add_common_layers_1D(y, BN_momentum=.95, dropout_rate=0):
    y = layers.LayerNormalization()(y)
    #y = layers.GaussianNoise(stddev=0.01)(y)
    #y = swish(y)
    y = layers.ReLU()(y)
    y = layers.SpatialDropout1D(dropout_rate)(y)
    return y


def add_common_layers_2D(y, BN_momentum=.95, dropout_rate=0):
    y = layers.LayerNormalization()(y)
    # y = layers.GaussianNoise(stddev=0.01)(y)
    y = swish(y)
    y = layers.SpatialDropout2D(dropout_rate)(y)
    return y

def add_common_layers_3D(y, BN_momentum=.95, dropout_rate=0):
    y = layers.BatchNormalization(momentum=BN_momentum)(y)
    y = layers.GaussianNoise(stddev=0.01)(y)
    y = swish(y)
    y = layers.SpatialDropout3D(dropout_rate)(y)
    return y

def dense_layer_1D(x, width, dropout_rate, layer_depth, BN_momentum=0.99):
    for layer_num in range(layer_depth):
        x = layers.Dense(units=width,
                         #padding='same',
                         kernel_initializer='he_normal')(x)
        x = add_common_layers_1D(x, BN_momentum=BN_momentum, dropout_rate=dropout_rate)
    return x


def RNN(x, init_filter_num=64, weight_regularization=0):
    x = layers.Bidirectional(layers.GRU(init_filter_num,
                                        return_sequences=True,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(weight_regularization),
                                        bias_regularizer=l2(weight_regularization),
                                        recurrent_regularizer=l2(weight_regularization)))(x)
    return x


def conv_layer_1D(y, num_channels, kernel_size=3, strides=1, dilation_rate=1, cardinality=1, weight_regularization=0):
    if cardinality == 1:
        return layers.Conv1D(num_channels,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             dilation_rate=dilation_rate,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_regularization),
                             bias_regularizer=l2(weight_regularization))(y)


    assert not num_channels % cardinality
    _d_out = num_channels // cardinality
    _d_in = y.shape[-1] // cardinality
    # _b_size = batch_size // cardinality
    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        # group = layers.Lambda(lambda z: z[j * _b_size:(j + 1) * _b_size, :, j * _d_in:(j + 1) * _d_in])(y)
        group = layers.Lambda(lambda z: z[:, :, j * _d_in:(j + 1) * _d_in])(y)
        groups.append(layers.Conv1D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides,
                                    padding='same', kernel_regularizer=l2(weight_regularization),
                                    bias_regularizer=l2(weight_regularization))(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)

    return y

def conv_chunks_layer_2D(y, num_channels, kernel_size=3, strides=(1, 1), dilation_rate=(1, 1), cardinality=1, weight_regularization=0):
    return layers.Conv2D(num_channels,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         dilation_rate=dilation_rate,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(weight_regularization),
                         bias_regularizer=l2(weight_regularization))(y)


def conv_layer_2D(y, num_channels, kernel_size, strides=(1, 1), dilation_rate=(1, 1), cardinality=1, weight_regularization=0):

    if cardinality == 1:
        return layers.Conv2D(num_channels,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             dilation_rate=dilation_rate,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_regularization),
                             bias_regularizer=l2(weight_regularization))(y)

    assert not num_channels % cardinality
    _d_out = num_channels // cardinality
    _d_in = y.shape[-1] // cardinality
    # _b_size = batch_size // cardinality
    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, j * _d_in:(j + 1) * _d_in])(y)
        groups.append(layers.Conv2D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides, padding='same',
                                    kernel_regularizer=l2(weight_regularization), bias_regularizer=l2(weight_regularization))(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)
    return y

def conv_chunks_layer_3D(y, num_channels, kernel_size, strides=(1, 1, 1), dilation_rate=(1, 1, 1), cardinality=1, weight_regularization=0):
    return layers.Conv3D(num_channels,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         dilation_rate=dilation_rate,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(weight_regularization),
                         bias_regularizer=l2(weight_regularization))(y)


def residual_block_1D(y, num_channels_in, num_channels, kernel_size, strides=1, _project_shortcut=False,
                      cardinality=1, bottle_neck=False, weight_regularization=0):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """

    shortcut = y

    if bottle_neck:
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv1D(num_channels_in, kernel_size=1, strides=1, padding='same')(y)
        y = add_common_layers_1D(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = conv_layer_1D(y, num_channels=num_channels_in, kernel_size=kernel_size, strides=1, cardinality=cardinality,
                          weight_regularization=weight_regularization)
        y = add_common_layers_1D(y)

        y = layers.Conv1D(num_channels, kernel_size=1, strides=1, padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = add_common_layers_1D(y)
    else:
        y = conv_layer_1D(y, num_channels=num_channels, kernel_size=kernel_size, strides=1, dilation_rate=1,
                          cardinality=cardinality, weight_regularization=weight_regularization)
        y = add_common_layers_1D(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut:
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv1D(num_channels, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = add_common_layers_1D(shortcut)

    y = layers.add([shortcut, y])
    return y


def residual_block_2D(y, num_channels_in, num_channels, kernel_size, strides=(1, 1), _project_shortcut=False,
                      cardinality=1, bottle_neck=False, weight_regularization=0):

    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """

    shortcut = y

    if bottle_neck:
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(num_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers_2D(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = conv_layer_2D(y, num_channels=num_channels_in, kernel_size=kernel_size, strides=1, cardinality=cardinality,
                          weight_regularization=weight_regularization)
        y = add_common_layers_2D(y)

        y = layers.Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = add_common_layers_2D(y)
    else:
        y = conv_layer_2D(y, num_channels=num_channels, kernel_size=kernel_size, strides=1, dilation_rate=(1, 1),
                          cardinality=cardinality, weight_regularization=weight_regularization)
        y = add_common_layers_2D(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut:
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(num_channels, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = add_common_layers_2D(shortcut)

    y = layers.add([shortcut, y])
    return y


def wavenet_residual_block_1D(y, num_channels, kernel_size, dilation_rate, weight_regularization=0):
    # Gated activation.

    l_sigmoid_conv1d = layers.Conv1D(num_channels,
                                     kernel_size,
                                     dilation_rate=dilation_rate,
                                     padding="same",
                                     activation="sigmoid",
                                     kernel_regularizer=l2(weight_regularization),
                                     bias_regularizer=l2(weight_regularization))(y)
    l_tanh_conv1d = layers.Conv1D(num_channels,
                                  kernel_size,
                                  dilation_rate=dilation_rate,
                                  padding="same",
                                  activation="tanh",
                                  kernel_regularizer=l2(weight_regularization),
                                  bias_regularizer=l2(weight_regularization))(y)
    l_mul = layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
    # Branches out to skip unit and residual output.
    l_skip_connection = layers.Conv1D(1, 1)(l_mul)
    l_residual = layers.Add()([y, l_skip_connection])
    return l_residual, l_skip_connection


def wavenet_residual_block_2D(y, num_channels, kernel_size, dilation_rate, weight_regularization=0):
    # Gated activation.
    l_sigmoid_conv1d = layers.Conv2D(num_channels,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation_rate,
                                     padding="same",
                                     activation="sigmoid",
                                     kernel_regularizer=l2(weight_regularization),
                                     bias_regularizer=l2(weight_regularization))(y)
    l_tanh_conv1d = layers.Conv2D(num_channels,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  padding="same",
                                  activation="tanh",
                                  kernel_regularizer=l2(weight_regularization),
                                  bias_regularizer=l2(weight_regularization))(y)
    l_mul = layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
    # Branches out to skip unit and residual output.
    l_skip_connection = layers.Conv2D(1, (1, 1))(l_mul)
    l_residual = layers.Add()([y, l_skip_connection])
    return l_residual, l_skip_connection