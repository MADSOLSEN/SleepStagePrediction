from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from ..blocks import add_common_layers_2D, conv_layer_2D, residual_block_2D, wavenet_residual_block_2D
import tensorflow.keras.backend as K
import tensorflow as tf

def cnn2d_model(input_shape,
                init_filter_num,
                kernel_size,
                num_layers,
                num_outputs,
                kernel_size_decrease_factor=1,
                number_of_classes=1,
                output_layer='sigmoid',
                kernel_size2=7,
                filter_increase_factor=1,
                layer_depth=1,
                dropout_rate=0,
                cardinality=1,
                bottle_neck=False,
                dilation_rate=1,
                BN_momentum=0.95,
                input_dropout=False,
                block_type='conv_1D',
                weight_regularization=0,
                chunk_time=False,
                block_size=30 * 4,
                block_stride=30,
                add_classifier=False):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    # test run if needed:
    #input_shape = [30 * 2 ** 3, 3]
    #x_input = gen_test_data(window=30 * 2 ** 3, batch_size=2, fs=1, num_features=3)
    #x = x_input

    if len(input_shape) == 2:
        x = layers.Reshape(target_shape=(input_shape + [1]))(x)

    if chunk_time:
        x = overlapping_blocker(x, block_size=block_size, stride=block_stride)
    else:
        x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    x = cnn2d(x,
              init_filter_num=init_filter_num,
              filter_increase_factor=filter_increase_factor,
              kernel_size=kernel_size,
              kernel_size_decrease_factor=kernel_size_decrease_factor,
              kernel_size2=kernel_size2,
              cardinality=cardinality,
              bottle_neck=bottle_neck,
              dropout_rate=dropout_rate,
              layer_depth=layer_depth,
              dilation_rate=dilation_rate,
              BN_momentum=BN_momentum,
              num_layers=num_layers,
              input_dropout=input_dropout,
              block_type=block_type,
              weight_regularization=weight_regularization)

    x = layers.Reshape((K.int_shape(x)[1], K.int_shape(x)[2] * K.int_shape(x)[3]))(x)

    if chunk_time:
        x = tf.reshape(x, [-1, num_outputs, x.shape[1] * x.shape[2]])

    if add_classifier:
        x = layers.Dense(units=number_of_classes,
                         activation=output_layer,
                         kernel_initializer='he_normal')(x)

    model = Model(inputs=x_input, outputs=x)
    return model


def zeropad(x, num_outputs, num_halfs):
    adjusted_input_size = num_outputs * 2 ** num_halfs
    if not (adjusted_input_size >= x.shape[1]):
        k = 1
    assert (adjusted_input_size >= x.shape[1])
    if adjusted_input_size > x.shape[1]:
        assert (adjusted_input_size / 2 == adjusted_input_size // 2)
        zeros_to_add = (adjusted_input_size - x.shape[1]) // 2
        x = layers.ZeroPadding2D(padding=(zeros_to_add, 0))(x) # only padding the time axis!

    return x


def overlapping_blocker(tensor, block_size, stride):

    x = tf.extract_volume_patches(tensor[..., None],
                                  ksizes=[1, block_size, 1, 1, 1],
                                  strides=[1, stride, 1, 1, 1],
                                  padding='SAME')
    x_reshaped = tf.reshape(x, [-1, tensor.shape[-2], block_size, tensor.shape[-1]])
    x_perm = tf.transpose(x_reshaped, perm=[0, 2, 1, 3])
    # x_reshaped = tf.reshape(x, [-1, block_size, tensor.shape[-2], tensor.shape[-1]])
    return x_perm


def cnn2d(x, block_type, init_filter_num=8, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
          layer_depth=1, num_layers=10, dilation_rate=1, BN_momentum=0.95, input_dropout=False, bottle_neck=False,
          weight_regularization=0, kernel_size2=3, kernel_size_decrease_factor=1):

    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)

    layer_counter = 0
    for n in range(num_layers):
        temporal_kernel_size = min(x.shape[1], kernel_size)
        kernel_size_ = (temporal_kernel_size, 1) if x.shape[-2] <= 1 else (temporal_kernel_size, kernel_size2)
        pool_size = (2, 1) if x.shape[-2] <= 1 else (2, 2)

        if block_type=='resnet':
            for m in range(layer_depth):
                if m == 0: y = x
                y = conv_layer_2D(y=y,
                                  dilation_rate=dilation_rate,
                                  num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                  kernel_size=kernel_size_,
                                  weight_regularization=weight_regularization)
                y = add_common_layers_2D(y, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
            y = layers.MaxPooling2D(pool_size=pool_size)(y)
            x = conv_layer_2D(y=x, dilation_rate=dilation_rate, strides=pool_size,
                              num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                              kernel_size=(1, 1), weight_regularization=weight_regularization)
            x = layers.add([x, y])
        else:
            for m in range(layer_depth):
                x = conv_layer_2D(y=x,
                                  dilation_rate=dilation_rate,
                                  num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                  kernel_size=kernel_size_,
                                  weight_regularization=weight_regularization)
                x = add_common_layers_2D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)

            x = layers.MaxPooling2D(pool_size=pool_size)(x)
        kernel_size = max(3, kernel_size // kernel_size_decrease_factor)
        layer_counter += 1
    return x

def gen_test_data(window=30 * 2 ** 3, batch_size=2, fs=32, num_features=1):

    # signal prep
    signal_batch = {
        signal_name: np.zeros((batch_size,
                               int(window * fs),
                               num_features)).astype('float32')
        for signal_name in ['PPG']
    }

    for bs in range(batch_size):
        signal = tf.constant(list(range(window * fs * num_features)))
        signal = tf.reshape(signal, [window * fs, num_features])
        signal_batch['PPG'][bs, :] = signal

    return tf.convert_to_tensor(signal_batch['PPG'])