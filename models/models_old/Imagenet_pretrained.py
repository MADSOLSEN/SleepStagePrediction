from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False)


def imagenet_pretrained_model(input_shape,
                              number_of_classes,
                              num_layers,
                              num_outputs,
                              dropout_rate,
                              layer_depth,
                              trainable=True,
                              add_Dense=False,
                              add_rnn=True,
                              add_classifier=False,
                              output_layer='sigmoid'):
    # input
    x_input = layers.Input(shape=np.array(input_shape))
    x = zeropad(x_input, num_outputs=num_outputs, num_halfs=num_layers)

    # Format input to match pre-trained model
    x = layers.Conv2D(3, kernel_size=1, strides=1, padding='same')(x)

    # model
    imagenet_model = VGG16(weights='imagenet', include_top=False)
    imagenet_model.trainable = trainable
    x = imagenet_model(x)

    if add_classifier:
        x = layers.Reshape((K.int_shape(x)[1], K.int_shape(x)[2] * K.int_shape(x)[3]))(x)

        num_class_adj = np.max((3, number_of_classes))
        pi = 1 / (num_class_adj)
        b = -np.log((1 - pi) / pi)

        if add_Dense:
            x = dense_layer(x,
                            width=512,
                            dropout_rate=dropout_rate,
                            layer_depth=layer_depth)

        if add_rnn:
            x = RNN(x,
                    init_filter_num=64,
                    dropout_rate=dropout_rate,
                    layer_depth=1)

        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)#,
                         #bias_initializer=tf.constant_initializer(b))(x)
    k = 1
    return Model(inputs=x_input, outputs=x)


def zeropad(x, num_outputs, num_halfs):
    adjusted_input_size = num_outputs * 2 ** num_halfs
    assert (adjusted_input_size >= x.shape[1])
    if adjusted_input_size > x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding2D(padding=(zeros_to_add, 0))(x) # only padding the time axis!
    return x

def dense_layer(x, width, dropout_rate, layer_depth):

    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        # y = layers.LeakyReLU()(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Dense(units=width,
                         # padding='same',
                         kernel_initializer='he_normal')(x)
        x = add_common_layers(x)
    return x

def RNN(x, init_filter_num=64, dropout_rate=0.5, layer_depth=1):
    def swish(x, beta=1):
        return (x * K.sigmoid(beta * x))

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = swish(y)
        y = layers.SpatialDropout1D(dropout_rate)(y)
        return y

    for layer_num in range(layer_depth):
        x = layers.Bidirectional(layers.GRU(init_filter_num,
                                            return_sequences=True,
                                            kernel_initializer="he_normal"))(x)

        x = add_common_layers(x)

    return x