from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def res_unet_model(input_shape,
                   number_of_classes,
                   init_filter_num,
                   cardinality,
                   dropout_rate,
                   kernel_size,
                   layer_depth,
                   num_layers_encoder,
                   num_layers_decoder,
                   num_outputs,
                   dilation_rate=1,
                   BN_momentum=0.99,
                   bottle_neck=False,
                   add_rnn=True,
                   add_classifier=True,
                   output_layer='sigmoid'):

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers_encoder - num_layers_decoder)

    # model
    x = res_unet(x,
                 init_filter_num=init_filter_num,
                 kernel_size=kernel_size,
                 cardinality=cardinality,
                 dropout_rate=dropout_rate,
                 layer_depth=layer_depth,
                 dilation_rate=dilation_rate,
                 BN_momentum=BN_momentum,
                 bottle_neck=bottle_neck,
                 num_layers_encoder=num_layers_encoder,
                 num_layers_decoder=num_layers_decoder)

    if add_rnn:
        x = RNN(x,
                init_filter_num=64,
                dropout_rate=dropout_rate,
                BN_momentum=BN_momentum,
                layer_depth=1)

    if add_classifier:
        num_class_adj = np.max((3, number_of_classes))
        pi = 1 / (num_class_adj)
        b = -np.log((1 - pi) / pi)
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)#,
                         #bias_initializer=tf.constant_initializer(b))(x)

    return Model(inputs=x_input, outputs=x)

def zeropad(x, num_outputs, num_halfs):

    adjusted_input_size = num_outputs * 2 ** num_halfs
    if adjusted_input_size != x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding1D(padding=zeros_to_add)(x)
    return x

def RNN(x, init_filter_num=64, dropout_rate=0.5, layer_depth=1, BN_momentum=0.99):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Bidirectional(layers.GRU(init_filter_num,
                                            return_sequences=True,
                                            kernel_initializer="he_normal"))(x)

        x = add_common_layers(x)

    return x


def res_unet(x, init_filter_num=16, kernel_size=3, cardinality=1, dropout_rate=0.5, layer_depth=2, num_layers_encoder=7,
             num_layers_decoder=2, dilation_rate=1, BN_momentum=0.99, bottle_neck=True):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization(momentum=BN_momentum)(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    def conv_layer(y, num_channels, kernel_size=3, strides=1, dilation_rate=1):

        if cardinality == 1:
            return layers.Conv1D(num_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer='he_normal')(y)

        assert not num_channels % cardinality
        _d_out = num_channels // cardinality
        _d_in = y.shape[-1] // cardinality
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, j * _d_in:j * _d_in + _d_in])(y)
            groups.append(layers.Conv1D(_d_out, dilation_rate=dilation_rate, kernel_size=kernel_size, strides=strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, kernel_size, _strides=1, _project_shortcut=False, bottle_neck=False):

        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """

        shortcut = y

        if bottle_neck:
            # we modify the residual building block as a bottleneck design to make the network more economical
            y = layers.Conv1D(nb_channels_in, kernel_size=1, strides=1, padding='same')(y)
            y = add_common_layers(y)

            # ResNeXt (identical to ResNet when `cardinality` == 1)
            y = conv_layer(y, num_channels=nb_channels_in, kernel_size=kernel_size, strides=1)
            y = add_common_layers(y)

            y = layers.Conv1D(nb_channels_out, kernel_size=1, strides=1, padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = add_common_layers(y)
        else:
            y = conv_layer(y, num_channels=nb_channels_out, kernel_size=kernel_size, strides=1, dilation_rate=1)
            y = add_common_layers(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut:
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv1D(nb_channels_out, kernel_size=1, strides=_strides, padding='same')(shortcut)
            shortcut = add_common_layers(shortcut)

        y = layers.add([shortcut, y])
        # TODO - consider adding activation here.
        return y


    d = {}
    d['conv0'] = conv_layer(y=x,
                            dilation_rate=dilation_rate,
                            num_channels=init_filter_num,
                            kernel_size=kernel_size)
    d['conv0'] = add_common_layers(d['conv0'])
    layer_counter = 1

    # ENCODER
    for n in range(num_layers_encoder):
        x = layers.MaxPooling1D(pool_size=2)(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            x = residual_block(y=x,
                               nb_channels_in=x.shape[-1],
                               nb_channels_out=init_filter_num * (layer_counter + 1),
                               kernel_size=kernel_size,
                               _strides=1,
                               _project_shortcut=True,
                               bottle_neck=bottle_neck)
            # x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1

    # DECODER
    for n in range(num_layers_decoder):
        x = layers.UpSampling1D(size=2)(d['conv{0}'.format(layer_counter - 1)])
        x = conv_layer(y=x, num_channels=init_filter_num * (num_layers_encoder - n), kernel_size=2)
        x = add_common_layers(x)
        x = layers.concatenate([d['conv{0}'.format(num_layers_encoder - (n + 1))], x], axis=2)
        for m in range(layer_depth):
            x = conv_layer(y=x,
                           dilation_rate=1,
                           num_channels=init_filter_num * (num_layers_encoder - n),
                           kernel_size=kernel_size)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1
    return d['conv{0}'.format(layer_counter - 1)]