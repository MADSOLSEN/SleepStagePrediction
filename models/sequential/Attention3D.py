from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def attention3d_model(input_shape,
                      number_of_classes,
                      num_outputs,
                      output_layer='sigmoid',
                      last_dropout_rate=0):

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    context_vector = attention_3d_block(x)
    context_vector = K.expand_dims(context_vector, axis=1)
    x = layers.Multiply()([x, context_vector])

    # Classification
    x = layers.SpatialDropout1D(last_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    k = 1
    return Model(inputs=x_input, outputs=x)

# This functions assumes that there is one specific output! Remember you have multiple timestep outputs.


def attention_3d_block(hidden_states):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    # # REMOVED pre_activation = concatenate([context_vector, h_t], name='attention_output')
    # # REMOVED attention_vector = Dense(hidden_states, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return context_vector

