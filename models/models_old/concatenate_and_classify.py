from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from models.sequential.Attention3D import Attention3D_block


def classification(models,
                   number_of_classes,
                   add_rnn=True,
                   add_attention=False,
                   output_layer='sigmoid',
                   dropout_rate=0,
                   last_dropout_rate=0,
                   BN_momentum=.95):

    if len(models)>1:
        concatenated = layers.concatenate([model.output for model in models], axis=-1)
    else:
        concatenated = models[0].output
    x = concatenated

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

    # Classifier
    x = layers.SpatialDropout1D(last_dropout_rate)(x)
    x = layers.Dense(units=number_of_classes,
                     activation=output_layer,
                     kernel_initializer='he_normal')(x)
    inputs = [model.input for model in models] # TODO - test this!
    return Model(inputs=inputs, outputs=x)

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

