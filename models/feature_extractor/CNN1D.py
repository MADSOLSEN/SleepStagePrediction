from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from models.blocks import add_common_layers_1D, conv_layer_1D, residual_block_1D, wavenet_residual_block_1D
import tensorflow as tf

def cnn1d_model(input_shape,
                init_filter_num,
                kernel_size,
                num_layers,
                num_outputs,
                number_of_classes=0,
                kernel_size_decrease_factor=1,
                output_layer='sigmoid',
                filter_increase_factor=1,
                dropout_rate=0,
                layer_depth=1,
                cardinality=1,
                bottle_neck=False,
                dilation_rate=1,
                BN_momentum=0.95,
                input_dropout=False,
                block_type='conv_1D',
                weight_regularization=0,
                chunk_time=False,
                block_size=30 * 4 * 32,
                block_stride=30 * 32,
                add_classifier=False):

    assert (filter_increase_factor == 1 or filter_increase_factor == 2)  # fixed or double.

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    # test run if needed:
    #input_shape = [30 * 2 ** 3, 3]
    #x_input = gen_test_data(window=30 * 2 ** 3, batch_size=2, fs=32, num_features=3)
    #x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)

    if chunk_time:
        x = overlapping_blocker(x, block_size=block_size, stride=block_stride)
    else:
        x = zeropad(x, num_outputs=num_outputs, num_halfs=num_layers)

    x = cnn1d(x,
              init_filter_num=init_filter_num,
              filter_increase_factor=filter_increase_factor,
              kernel_size=kernel_size,
              kernel_size_decrease_factor=kernel_size_decrease_factor,
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

    if chunk_time:
        #x_temp0 = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
        x = tf.reshape(x, [-1, num_outputs, x.shape[1] * x.shape[2]])
        K = 1
    if add_classifier:
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

def overlapping_blocker(tensor, block_size, stride):

    test = False
    if test:

        block_size_ = 8
        stride_ = 4
        num_outputs = 8

        # test with knows sizes:
        input1 = tf.constant(list(range(32 * 1)))
        input2 = tf.constant(list(range(32 * 1)))
        input1 = tf.reshape(input1, [1, 32, 1])
        input2 = tf.reshape(input2, [1, 32, 1])
        input = tf.concat([input1, input2], axis=0)
        x = tf.extract_volume_patches(input[..., None, None],
                                      ksizes=[1, block_size_, 1, 1, 1],
                                      strides=[1, stride_, 1, 1, 1],
                                      padding='SAME')
        x_reshaped = tf.reshape(x, [-1, block_size_, input.shape[-1]])

        # model
        y = layers.MaxPooling1D(pool_size=2)(x_reshaped)

        # reshape output:
        # x_temp0 = tf.reshape(y, [-1, y.shape[1] * y.shape[2]])
        x_out = tf.reshape(x, [-1, num_outputs, y.shape[1] * y.shape[2]])
        k = 1

    k = 1
    x = tf.extract_volume_patches(tensor[..., None, None],
                                  ksizes=[1, block_size, 1, 1, 1],
                                  strides=[1, stride, 1, 1, 1],
                                  padding='SAME')
    x_reshaped = tf.reshape(x, [-1, tensor.shape[-1], block_size])
    x_perm = tf.transpose(x_reshaped, perm=[0, 2, 1])

    # x_reshaped = tf.reshape(x, [-1, block_size, tensor.shape[-1]])
    return x_perm

def cnn1d(x, block_type, init_filter_num=8, filter_increase_factor=1, kernel_size=3, cardinality=1, dropout_rate=0,
          layer_depth=1, num_layers=10, dilation_rate=1, BN_momentum=0.95, input_dropout=False, bottle_neck=False,
          weight_regularization=0, kernel_size_decrease_factor=1):

    if input_dropout:
        x = layers.SpatialDropout1D(1/x.shape[-1])(x)

    layer_counter = 0
    for n in range(num_layers):
        kernel_size_ = min(x.shape[1], kernel_size)

        if block_type == 'resnet':
            for m in range(layer_depth):
                if m == 0: y = x
                y = conv_layer_1D(y=y,
                                  dilation_rate=dilation_rate,
                                  num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                  kernel_size=kernel_size_,
                                  weight_regularization=weight_regularization)
                y = add_common_layers_1D(y, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
            y = layers.MaxPooling1D(pool_size=2)(y)
            x = conv_layer_1D(y=x,
                              dilation_rate=dilation_rate,
                              num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                              kernel_size=1,
                              strides=2,
                              weight_regularization=weight_regularization)
            x = layers.add([x, y])
        else:
            for m in range(layer_depth):
                x = conv_layer_1D(y=x,
                                  dilation_rate=dilation_rate,
                                  num_channels=init_filter_num * filter_increase_factor ** layer_counter,
                                  kernel_size=kernel_size_,
                                  weight_regularization=weight_regularization)
                x = add_common_layers_1D(x, dropout_rate=dropout_rate, BN_momentum=BN_momentum)
            x = layers.MaxPooling1D(pool_size=2)(x)
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
