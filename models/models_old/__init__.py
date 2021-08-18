from .unet import unet_model
from .utils import get_callbacks, reconfigure_model, get_model_activation, predict_dataset, train, predict_dataset_semantic, concatenate_models, concatenate_model_outputs, visualize_model, stack_models
from .CNN2D import cnn2D_model
from .Imagenet_pretrained import imagenet_pretrained_model
from .YAMnet import YAMnet_model
from .ecg_rnn import ecg_rnn_model
from .res_unet import res_unet_model
from .dense import dense_model
from .stan import stan_model
from .unet_spatial_filter import unet_spa_model
from .Attention3D import Attention3D_block
from .concatenate_and_classify import classification
from .classifier import classifier_model
from .blocks import add_common_layers_1D, add_common_layers_2D, dense_layer_1D, RNN, conv_layer_1D, conv_layer_2D, residual_block_1D, residual_block_2D, wavenet_residual_block_1D, wavenet_residual_block_2D

__all__ = [
    'unet_model',
    'res_unet_model',
    'imagenet_pretrained_model',
    'CNN2D',
    'get_callbacks',
    'reconfigure_model',
    'get_model_activation',
    'predict_dataset',
    'predict_dataset_semantic',
    'train',
    'classification',
    'concatenate_models',
    'concatenate_model_outputs',
    'visualize_model',
    'stack_models',
    'dense_model',
    'stan_model',
    'unet_spa_model',
    'Attention3D_block',
    'classifier_model'
]

models = {
    'unet_model': unet_model,
    'res_unet_model': res_unet_model,
    'cnn2D_model': cnn2D_model,
    'imagenet_pretrained_model': imagenet_pretrained_model,
    'YAMnet_model': YAMnet_model,
    'ecg': ecg_rnn_model,
    'dense_model': dense_model,
    'stan_model': stan_model,
    'unet_spa_model': unet_spa_model,
    'classifier_model': classifier_model
}

blocks = {
    'add_common_layers_1D': add_common_layers_1D,
    'add_common_layers_2D': add_common_layers_2D,
    'dense_layer_1D': dense_layer_1D,
    'RNN': RNN,
    'conv_layer_1D': conv_layer_1D,
    'conv_layer_2D': conv_layer_2D,
    'residual_block_1D': residual_block_1D,
    'residual_block_2D': residual_block_2D,
    'wavenet_residual_block_1D': wavenet_residual_block_1D,
    'wavenet_residual_block_2D': wavenet_residual_block_1D
}

# TODO - Step 1: Find best filter width and kernel size configuration maybe include dilation.
# TODO - step 2: Choose block structure given the above.
# TODO - step 3: Choose best default vs. 3xLSTM (HRV paper) vs. Unet vs. wavenet dilation, attention
# TODO - scale model
# TODO - augmentation, dropout, input...


# TODO MUST - #1 do receptive field first - only kernel size and dilation with filter width maybe.
# TODO MUST - #2 do block structure. This could be the first as well.. Think about this
# TODO MUST - #3 do default vs. 3xLSTM (HRV paper) vs. Unet vs. wavenet dilation, attention
# TODO - #4 model scale, augmentation, dropout,
# TODO - #5 how do we know if some results are poor, that they are not just overfitted, and would be better if we include dropout.
# TODO - #5.1 how about including dropout during these training runs, just to be certain that it is not overfitting.
# TODO - compare your model feature extraction with feature based extraction with sequential layer on top.