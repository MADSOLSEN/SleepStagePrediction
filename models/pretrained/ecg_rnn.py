from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np


def ecg_rnn_model(input_shape,
                  num_outputs,
                  number_of_classes=2,
                  rnn_filter_num=128,
                  dense_filter_num=512,
                  dropout_rate=0.5,
                  BN_momentum=0.99
                  ):
    
    # input
    x_input = layers.Input(shape=tuple(input_shape))

    # model
    x = ecg_rnn_network(x_input,
                        rnn_filter_num=rnn_filter_num,
                        dense_filter_num=dense_filter_num,
                        dropout_rate=dropout_rate,
                        BN_momentum=BN_momentum)

    # Softmax output:
    outputs = layers.Dense(units=number_of_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(x)

    return Model(inputs=x_input, outputs=outputs)

def ecg_rnn_network(x,
                    rnn_filter_num,
                    dense_filter_num,
                    dropout_rate,
                    BN_momentum):

    x = layers.Bidirectional(layers.GRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)

    x = layers.Bidirectional(layers.GRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)
    x = layers.BatchNormalization(momentum=BN_momentum)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.GRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)
    x = layers.Bidirectional(layers.GRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)

    x = layers.BatchNormalization(momentum=BN_momentum)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_filter_num,
              activation='relu')(x)

    return x


def ecg_rnn_network_old(x,
                    rnn_filter_num,
                    dense_filter_num,
                    dropout_rate,
                    BN_momentum):

    x = layers.Bidirectional(layers.CuDNNGRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)

    x = layers.Bidirectional(layers.CuDNNGRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)
    x = layers.BatchNormalization(momentum=BN_momentum)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.CuDNNGRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)
    x = layers.Bidirectional(layers.CuDNNGRU(rnn_filter_num,
                               return_sequences=True,
                               kernel_initializer="he_normal"))(x)

    x = layers.BatchNormalization(momentum=BN_momentum)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_filter_num,
              activation='relu')(x)

    return x