from functions.metrics import f1_function, precision_function, recall_function, get_false_negative_events, \
    get_false_positive_events, get_true_positive_events, MCC, accuracy_semantic_input_function, \
    recall_semantic_input_function, precision_semantic_input_function, specificity_semantic_input_function, \
    precision_mean, recall_mean, f1_mean_metric, specificity_mean, cohens_kappa, optimized_f1, calculate_AUCRC,  \
    calculate_AUROC, optimized_f12, cohens_kappa1, binary_accuracy, balanced_accuracy, balanced_accuracy_new, \
    f1_by_class_metric, absolute_error, false_negative_function, true_positives_function, false_positives_function

from functions.loss import cross_entropy_loss, weighted_loss, binary_focal_loss, binary_focal_loss_weighted, \
    categorical_focal_loss, categorical_focal_loss_weighted, effective_number_binary_focal_loss, \
    effective_number_cross_entropy_loss, categorical_crossentropy_loss, categorical_crossentropy_weighted_loss, \
    OHEM_cross_entropy_loss, OHEM_cross_entropy_weighted_loss, OHEM_cross_entropy_efficient_number_loss, \
    OHNM_cross_entropy_loss, OHNM_cross_entropy_weighted_loss, OHNM_cross_entropy_efficient_number_loss, \
    DICED_LOSS

from functions.learning_rate import OSCILLATOR_EXP_DECAY

learning_rate_schedules = {
    'oscillator_exp_decay': OSCILLATOR_EXP_DECAY
}

loss_functions = {
    'cross_entropy_loss': cross_entropy_loss, #
    'weighted_loss': weighted_loss,  #
    'binary_focal_loss': binary_focal_loss, #
    'binary_focal_loss_weighted': binary_focal_loss_weighted, #
    'effective_number_binary_focal_loss': effective_number_binary_focal_loss, #
    'effective_number_cross_entropy_loss': effective_number_cross_entropy_loss, #
    'ohem_cross_entropy_loss': OHEM_cross_entropy_loss,
    'ohem_cross_entropy_weighted_loss': OHEM_cross_entropy_weighted_loss,
    'ohem_cross_entropy_efficient_number_loss': OHEM_cross_entropy_efficient_number_loss,
    'ohnm_cross_entropy_loss': OHNM_cross_entropy_loss,
    'ohnm_cross_entropy_weighted_loss': OHNM_cross_entropy_weighted_loss,
    'ohnm_cross_entropy_efficient_number_loss': OHNM_cross_entropy_efficient_number_loss,
    'categorical_focal_loss': categorical_focal_loss,
    'categorical_focal_loss_weighted': categorical_focal_loss_weighted,
    'categorical_crossentropy_loss': categorical_crossentropy_loss,
    'categorical_crossentropy_weighted_loss': categorical_crossentropy_weighted_loss,
    'DICED_LOSS': DICED_LOSS
}

metric_functions = {
    'f1': f1_function(),
    'precision': precision_function(),
    'recall': recall_function(),
    'true_positives': true_positives_function(),
    'false_positives': false_positives_function(),
    'false_negatives': false_negative_function(),
    'get_false_negative_events': get_false_negative_events,
    'get_false_positive_events': get_false_positive_events,
    'get_true_positive_events': get_true_positive_events,
    'accuracy_semantic_input': accuracy_semantic_input_function(),
    'precision_semantic_input': precision_semantic_input_function(),
    'recall_semantic_input': recall_semantic_input_function(),
    'specificity_semantic_input': specificity_semantic_input_function(),
    'AUCRC': calculate_AUCRC,
    'AUROC': calculate_AUROC,
    'absolute_error': absolute_error()
}

metrics = {
    'binary_accuracy': binary_accuracy,
    'balanced_accuracy': balanced_accuracy,
    'balanced_accuracy_new': balanced_accuracy_new,
    'AUCRC': calculate_AUCRC,
    'AUROC': calculate_AUROC,
    'f1_mean': f1_mean_metric,
    'f1_by_class': f1_by_class_metric,
    'recall': recall_mean,
    'precision': precision_mean,
    'specificity': specificity_mean,
    'MCC': MCC,
    'cohens_kappa': cohens_kappa,
    'cohens_kappa1': cohens_kappa1,
    'optimized_f1': optimized_f1,
    'optimized_f12': optimized_f12,
    'absolute_error': absolute_error
}

