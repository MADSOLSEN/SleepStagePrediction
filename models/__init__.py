from .feature_extractor import dense_model, cnn1d_model, cnn2d_model, cnn1d_spatial_model
from .sequential import classifier_model, attention3d_model, recurrent_model, unet_model, wavenet_model, unet_spatial_model, papini_model
from .pretrained import imagenet_pretrained_model, YAMnet_model
from .blocks import add_common_layers_1D, add_common_layers_2D, dense_layer_1D, RNN, conv_layer_1D, conv_layer_2D, \
    residual_block_1D, residual_block_2D, wavenet_residual_block_1D, wavenet_residual_block_2D, conv_chunks_layer_2D, \
    conv_chunks_layer_3D, add_common_layers_3D
from .blocks_downloaded import CausalAtrousConvolution1D
from .utils import get_callbacks, reconfigure_model, get_model_activation, predict_dataset, train, \
    predict_dataset_semantic, concatenate_models, concatenate_model_outputs, visualize_model, stack_models, compute_receptive_field
from .full_stack_models import unet1d_model, sridhar_model, ecg_rnn_model, cnn1d_full_model, wavenet_full_model, cnn_recurrent_model, cnn_dense_model
from models.full_stack_models.unet.lixiaolei import up_and_concate, attention_block_2d, attention_up_and_concate, rec_res_block, res_block
from models.full_stack_models.unet import unet_model_li, att_unet_model_li, r2_unet_model_li, att_r2_unet_model_li, att_r2_unet_aux_model_li, USleep_Att
from models.threshold import treshold_model as threshold

__all__ = [
    'dense_model',
    'cnn1d_model',
    'cnn2d_model',
    'classifier_model',
    'attention3d_model',
    'recurrent_model',
    'unet_model',
    'wavenet_model',
    'ecg_rnn_model',
    'imagenet_pretrained_model',
    'YAMnet_model',
    'add_common_layers_1D',
    'add_common_layers_2D',
    'dense_layer_1D',
    'RNN',
    'conv_layer_1D',
    'conv_layer_2D',
    'residual_block_1D',
    'residual_block_2D',
    'wavenet_residual_block_1D',
    'wavenet_residual_block_2D',
    'CausalAtrousConvolution1D',
    'get_callbacks',
    'reconfigure_model',
    'get_model_activation',
    'predict_dataset',
    'train',
    'predict_dataset_semantic',
    'concatenate_models',
    'concatenate_model_outputs',
    'visualize_model',
    'stack_models',
    'compute_receptive_field',
    'conv_chunks_layer_2D',
    'conv_chunks_layer_3D',
    'cnn1d_full_model',
    'sridhar_model',
    'USleep_Att',
    'cnn_recurrent_model',
    'cnn_dense_model'
]

models = {
    'dense_model': dense_model,
    'cnn1d_model': cnn1d_model,
    'cnn2d_model': cnn2d_model,
    'cnn1d_spatial_model': cnn1d_spatial_model,
    'classifier_model': classifier_model,
    'attention3d_model': attention3d_model,
    'recurrent_model': recurrent_model,
    'unet_model': unet_model,
    'wavenet_model': wavenet_model,
    'ecg_rnn_model': ecg_rnn_model,
    'imagenet_pretrained_model': imagenet_pretrained_model,
    'YAMnet_model': YAMnet_model,
    'unet_spatial_model': unet_spatial_model,
    'papini_model': papini_model,
    'unet1d_model': unet1d_model,
    'sridhar_model': sridhar_model,
    'cnn1d_full_model': cnn1d_full_model,
    'unet_model_li': unet_model_li,
    'att_unet_model_li': att_unet_model_li,
    'r2_unet_model_li': r2_unet_model_li,
    'att_r2_unet_model_li': att_r2_unet_model_li,
    'wavenet_full_model': wavenet_full_model,
    'att_r2_unet_aux_model_li': att_r2_unet_aux_model_li,
    'USleep_Att': USleep_Att,
    'threshold': threshold,
    'cnn_recurrent': cnn_recurrent_model,
    'cnn_dense': cnn_dense_model
}

