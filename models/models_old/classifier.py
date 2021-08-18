from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

def classifier_model(input_shape,
                     number_of_classes,
                     num_outputs,
                     batch_size,
                     output_layer='sigmoid'):

    x_input = layers.Input(shape=np.array(input_shape))
    x = x_input

    if len(input_shape) == 3:
        x = layers.Reshape((x_input.shape[1], x_input.shape[2] * x_input.shape[3]))(x)

    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    model = Model(inputs=x_input, outputs=x)
    return model
