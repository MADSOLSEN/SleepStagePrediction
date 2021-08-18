from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K
from models.sequential.Attention3D import Attention3D_block

def unet_model(input_shape,
               batch_size,
               number_of_classes,
               init_filter_num,
               filter_increase_factor,
               cardinality,
               dropout_rate,
               kernel_size,
               layer_depth,
               num_layers_encoder,
               num_layers_decoder,
               num_outputs,
               last_dropout_rate=0,
               dilation_rate=1,
               BN_momentum=0.95,
               add_rnn=True,
               add_attention=False,
               add_classifier=False,
               output_layer='sigmoid',
               input_dropout=False,
               resnet=False,
               bottle_neck=False,
               wavenet=False):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)
    x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers_encoder - num_layers_decoder)

    # model
    x = unet(x,
             init_filter_num=init_filter_num,
             filter_increase_factor=filter_increase_factor,
             batch_size=batch_size,
             kernel_size=kernel_size,
             cardinality=cardinality,
             dropout_rate=dropout_rate,
             layer_depth=layer_depth,
             dilation_rate=dilation_rate,
             BN_momentum=BN_momentum,
             num_layers_encoder=num_layers_encoder,
             num_layers_decoder=num_layers_decoder,
             input_dropout=input_dropout,
             resnet=resnet,
             bottle_neck=bottle_neck,
             wavenet=wavenet)

    if add_rnn:
        if x.shape[1] != 1:
            x = RNN(x,
                    init_filter_num=64,
                    dropout_rate=dropout_rate,
                    BN_momentum=BN_momentum,
                    layer_depth=1)

    if add_attention:
        # x_ = layers.Permute((2, 1))(x)
        attention_output = Attention3D_block(x)
        x = layers.Multiply()([x, attention_output])

    if add_classifier:
        x = layers.SpatialDropout1D(last_dropout_rate)(x)
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)
    model = Model(inputs=x_input, outputs=x)
    return model

def zeropad(x, num_outputs, num_halfs):
    if num_halfs < 0:
        return x
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


def unet(x, init_filter_num=16, batch_size=32, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
         layer_depth=2, num_layers_encoder=7, num_layers_decoder=2, dilation_rate=1, BN_momentum=0.95,
         input_dropout=False, resnet=False, bottle_neck=False, wavenet=False):

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
        #_b_size = batch_size // cardinality
        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            #group = layers.Lambda(lambda z: z[j * _b_size:(j + 1) * _b_size, :, j * _d_in:(j + 1) * _d_in])(y)
            group = layers.Lambda(lambda z: z[:, :, j * _d_in:(j + 1) * _d_in])(y)
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
        return y

    def wavenet_residual_block(l_input, num_filters, kernel_size, dilation_rate):
        # Gated activation.

        l_sigmoid_conv1d = layers.Conv1D(num_filters,
                                         kernel_size,
                                         dilation_rate=dilation_rate,
                                         padding="same",
                                         activation="sigmoid")(l_input)
        l_tanh_conv1d = layers.Conv1D(num_filters,
                                      kernel_size,
                                      dilation_rate=dilation_rate,
                                      padding="same",
                                      activation="tanh")(l_input)
        l_mul = layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
        # Branches out to skip unit and residual output.
        l_skip_connection = layers.Conv1D(1, 1)(l_mul)
        l_residual = layers.Add()([l_input, l_skip_connection])
        return l_residual, l_skip_connection

    d = {}
    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)
    #x = conv_layer(y=x,
    #               dilation_rate=dilation_rate,
    #               num_channels=init_filter_num,
    #               kernel_size=kernel_size)

    if num_layers_encoder > 0:
        x = layers.Conv1D(init_filter_num,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          dilation_rate=dilation_rate,
                          kernel_initializer='he_normal')(x)

        d['conv0'] = add_common_layers(x)
    layer_counter = 1

    # ENCODER
    # TODO - in the final experiments. kernel size of u-net should max be temporal length.
    for n in range(num_layers_encoder):
        kernel_size_ = min(x.shape[1], kernel_size)

        x = layers.MaxPooling1D(pool_size=2)(d['conv{0}'.format(layer_counter - 1)])
        for m in range(layer_depth):
            k = 1
            if resnet:
                project_shortcut = True if m == 0 else False
                x = residual_block(y=x,
                                   nb_channels_in=x.shape[-1],
                                   nb_channels_out=init_filter_num * filter_increase_factor ** layer_counter,
                                   kernel_size=kernel_size_,
                                   _strides=1,
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

    # DECODER
    for n in range(num_layers_decoder):
        x = layers.UpSampling1D(size=2)(d['conv{0}'.format(layer_counter - 1)])

        x = conv_layer(y=x, num_channels=x.shape[-1] // 2 , kernel_size=2)
        x = layers.concatenate([d['conv{0}'.format(num_layers_encoder - (n + 1))], x], axis=2)
        kernel_size_ = min(x.shape[1], kernel_size)

        for m in range(layer_depth):
            k = 1
            if resnet:
                project_shortcut = True if m == 0 else False
                x = residual_block(y=x,
                                   nb_channels_in=x.shape[-1],
                                   nb_channels_out=init_filter_num * filter_increase_factor ** (num_layers_encoder - (n + 1)),
                                   kernel_size=kernel_size_,
                                   _strides=1,
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
                               num_channels=init_filter_num * filter_increase_factor ** (num_layers_encoder - (n + 1)),
                               kernel_size=kernel_size_)
            x = add_common_layers(x)
        d['conv{0}'.format(layer_counter)] = x
        layer_counter += 1
    return x
