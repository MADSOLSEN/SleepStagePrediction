from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def treshold_model(input_shape,
                   number_of_classes,
                   num_outputs,
                   output_layer='softmax',
                   add_dense=False,
                   init_filter_num=128,
                   filter_increase_factor=1,
                   num_layers=1,
                   layer_depth=1,
                   dropout_rate=0,
                   BN_momentum=0.95,
                   maxpool=True,
                   weight_regularization=0,
                   input_dropout_rate=0,
                   output_dropout_rate=0,
                   threshold=20):

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input


    # layers.ThresholdedReLU(theta=threshold)
    x = layers.Dense(units=1,
                     kernel_initializer='he_normal')(x)
    #x = layers.BatchNormalization()(x)

    #x = K.greater_equal(x, threshold)
    # x = layers.Softmax(axis=-1)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    model = Model(inputs=x_input, outputs=x)
    return model

