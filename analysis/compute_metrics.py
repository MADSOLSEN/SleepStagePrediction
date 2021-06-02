import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, cohen_kappa_score, accuracy_score
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
from functions import metric_functions
from models import predict_dataset, predict_dataset_semantic
from signal_processing import post_processing, preprocessing
from utils import binary_to_array, CSVHandler, inverse_events, intersection_overlap

def compute_semantic_performance_argmax(model,
                                        test_dataset,
                                        sleep_metrics=True,
                                        apply_mask=True):

        epsilon = sys.float_info.epsilon
        events = [event['name'] for event in (test_dataset.events_format)]

        biometrics = ['age', 'sex', 'bmi']  # TODO - later inside the dataset itself
        sleep_events = ['SDB', 'arousal', 'PLM', 'TST']
        performance_labels = ['N', 'tn', 'fp', 'fn', 'tp', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1',
                              'specificity', 'cohens_kappa', 'MCC',
                              'number_of_events_tar', 'number_of_events_pre']

        performance_multiclass = {label: [] for label in ['recordings', 'accuracy', 'cohens_kappa']}
        performance = {label: [] for label in ['recordings', 'events'] + performance_labels}
        record_based_out = {label: [] for label in ['recordings', 'rec_len', 'N_masks'] + sleep_events + biometrics}

        output = {
            record: {
                event: {
                    vec: []
                    for vec in ['tar', 'pre', 'pre_bin']
                } for event in events + ['mask']
            } for record in test_dataset.records}


        if sleep_metrics:
            sleep_labels = ['tst', 'sleep_latency', 'waso', 'noa5', 'se'] + [e for e in events if
                                                                             e in ['light', 'deep', 'rem']] + \
                           ['{}_fraction'.format(e) for e in events if e in ['light', 'deep', 'rem']]
            sleep_metrics_out = ({'recordings': []})
            sleep_metrics_out.update(
                {'{}_{}'.format(label, type): [] for type in ['tar', 'pre_bin'] for label in sleep_labels})
            record_based_out.update(sleep_metrics_out)

        # predict dataset_semantic
        prediction, target = predict_dataset_semantic(
            model=model,
            dataset=test_dataset)


        for record in test_dataset.records:

            if len([t for event in events for t in target[record][event] if t > -.5]) <= 0:
                print(record)
                assert(len([t for event in events for t in target[record][event] if t > -.5]) > 0)

            tar = np.argmax([tar for tar in target[record].values()], axis=0)
            pre = np.argmax([pre for pre in prediction[record].values()], axis=0)


            if apply_mask:
                mask = np.array([(tar == -1) * 1 for tar in target[record][events[0]]])
                tar = tar[mask == 0]
                pre = pre[mask == 0]

            # multiclass
            performance_multiclass['recordings'].append(record)
            performance_multiclass['accuracy'].append(accuracy_score(y_true=tar, y_pred=pre))
            performance_multiclass['cohens_kappa'].append(cohen_kappa_score(y1=tar, y2=pre))


            for event_num, event in enumerate(events):

                tar_bin = (1 * (tar == event_num))
                pre_bin = (1 * (pre == event_num))

                tn, fp, fn, tp = confusion_matrix(y_true=list(tar_bin) + [1], y_pred=list(pre_bin) + [0]).ravel()
                fn -= 1

                performance['recordings'].append(record)
                performance['events'].append(event)

                performance['N'].append(tp + fp + fn + tn)
                performance['tp'].append(tp)
                performance['fp'].append(fp)
                performance['tn'].append(tn)
                performance['fn'].append(fn)

                performance['accuracy'].append((tp + tn) / (tn + fp + fn + tp + epsilon))
                performance['balanced_accuracy'].append((tp / (tp + fn + epsilon) + tn / (tn + fp + epsilon)) / 2)

                performance['recall'].append((tp) / (tp + fn + epsilon))
                performance['precision'].append((tp) / (fp + tp + epsilon))
                performance['specificity'].append((tn) / (tn + fp + epsilon))
                performance['f1'].append(2 * ((tp) / (fp + tp + epsilon) * (tp) / (tp + fn + epsilon)) / ((tp) / (fp + tp + epsilon) + (tp) / (tp + fn + epsilon) + epsilon))
                performance['cohens_kappa'].append(1 - (1 - (tp + tn) / (tp + tn + fp + fn + epsilon)) / (1 - ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn + epsilon) ** 2 + epsilon))
                performance['MCC'].append((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** (1 / 2) + epsilon))


                performance['number_of_events_tar'].append(tp + fn)
                performance['number_of_events_pre'].append(tp + fp)

                # output
                output[record][event]['tar'] = list(tar_bin)
                output[record][event]['pre'] = list(prediction[record][event])
                output[record][event]['pre_bin'] = list(pre_bin)
            output[record]['mask']['tar'] = list(mask)

            # Record based out
            record_based_out['recordings'].append(record)
            record_based_out['rec_len'].append(tar.shape[0])
            record_based_out['N_masks'].append(mask.sum())
            record_based_out['SDB'].append(test_dataset.get_number_of_events(record=record, event_label='SDB'))
            record_based_out['arousal'].append(test_dataset.get_number_of_events(record=record, event_label='arousal'))
            record_based_out['PLM'].append(test_dataset.get_number_of_events(record=record, event_label='PLM'))
            record_based_out['TST'].append(test_dataset.get_TST(record=record))

            record_based_out['age'].append(get_biometrics(record=record, metric='age'))
            record_based_out['sex'].append(get_biometrics(record=record, metric='sex'))
            record_based_out['bmi'].append(get_biometrics(record=record, metric='bmi'))


            if sleep_metrics:
                for type in ['tar', 'pre_bin']:
                    if ('rem' in events) & ('deep' in events) & ('light' in events) & ('wake' in events):
                        sleep = np.array(binary_to_array([abs(o - 1) for o in output[record]['wake'][type]]))
                        wake = np.array(binary_to_array(output[record]['wake'][type]))
                        light = np.array(binary_to_array(output[record]['light'][type]))
                        deep = np.array(binary_to_array(output[record]['deep'][type]))
                        rem = np.array(binary_to_array(output[record]['rem'][type]))

                    else: # ('sleep' in events) & ('wake' in events):
                        sleep = np.array(binary_to_array(output[record]['sleep'][type]))
                        wake = np.array(binary_to_array(output[record]['wake'][type]))
                        light = []
                        deep = []
                        rem = []

                    do = False
                    if len(sleep > 0):
                        if any((sleep[:, 1] - sleep[:, 0]) >= 5 / (test_dataset.prediction_resolution / 60)):
                            do = True

                    if do:
                        # valid sleep period
                        sl_thresh = (sleep[:, 1] - sleep[:, 0]) >= 5 / (test_dataset.prediction_resolution / 60)
                        sl_min = np.min(sleep[sl_thresh, 0]) if len(sleep[:, 0]) > 0 else np.nan
                        sl_max = np.max(sleep[sl_thresh, 1]) if len(sleep) > 0 else np.nan

                        sleep_idx = (sleep[:, 0] >= sl_min) & (sleep[:, 0] <= sl_max) if len(sleep) > 0 else False
                        tst = np.sum(sleep[sleep_idx, 1] - sleep[sleep_idx, 0]) if len(sleep[sleep_idx]) > 0 else np.nan


                        spt = sl_max - sl_min if len(sleep) > 0 else np.nan

                        # indexes
                        wake_idx = (wake[:, 0] >= sl_min) & (wake[:, 1] <= sl_max) if len(wake) > 0 else False
                        light_idx = (light[:, 0] >= sl_min) & (light[:, 1] <= sl_max) if len(light) > 0 else False
                        deep_idx = (deep[:, 0] >= sl_min) & (deep[:, 1] <= sl_max) if len(deep) > 0 else False
                        rem_idx = (rem[:, 0] >= sl_min) & (rem[:, 1] <= sl_max) if len(rem) > 0 else False

                        # calculate metrics
                        waso = np.sum(wake[wake_idx, 1] - wake[wake_idx, 0]) if len(wake[wake_idx]) > 0 else np.nan
                        se = 1 - (waso / (waso + tst + epsilon)) if waso and tst else np.nan
                        noa5 = np.sum((wake[wake_idx, 1] - wake[
                            wake_idx, 0]) * test_dataset.prediction_resolution / 60 > 5) if len(
                            wake[wake_idx]) > 0 else np.nan

                        record_based_out['tst_' + type].append(tst * test_dataset.prediction_resolution / 60)
                        record_based_out['sleep_latency_' + type].append(
                            sl_min * test_dataset.prediction_resolution / 60)
                        record_based_out['waso_' + type].append(waso * test_dataset.prediction_resolution / 60)
                        record_based_out['noa5_' + type].append(noa5)
                        record_based_out['se_' + type].append(se)

                        if 'light' in events:
                            if (len(light) > 0):
                                laso = np.sum(light[light_idx, 1] - light[light_idx, 0]) if len(
                                    light[light_idx]) > 0 else np.nan
                                record_based_out['light_' + type].append(laso * test_dataset.prediction_resolution / 60)
                                record_based_out['light_fraction_' + type].append(laso / spt)
                            else:
                                record_based_out['light_' + type].append(0)
                                record_based_out['light_fraction_' + type].append(0)
                        if 'deep' in events:
                            if (len(deep) > 0):
                                daso = np.sum(deep[deep_idx, 1] - deep[deep_idx, 0]) if len(
                                    deep[deep_idx]) > 0 else np.nan
                                record_based_out['deep_' + type].append(daso * test_dataset.prediction_resolution / 60)
                                record_based_out['deep_fraction_' + type].append(daso / spt)
                            else:
                                record_based_out['deep_' + type].append(0)
                                record_based_out['deep_fraction_' + type].append(0)
                        if 'rem' in events:
                            if (len(rem) > 0):
                                raso = np.sum(rem[rem_idx, 1] - rem[rem_idx, 0]) if len(rem[rem_idx]) > 0 else np.nan
                                record_based_out['rem_' + type].append(raso * test_dataset.prediction_resolution / 60)
                                record_based_out['rem_fraction_' + type].append(raso / spt)
                            else:
                                record_based_out['rem_' + type].append(0)
                                record_based_out['rem_fraction_' + type].append(0)
                    else:
                        record_based_out['tst_' + type].append(0)
                        record_based_out['sleep_latency_' + type].append(0)
                        record_based_out['waso_' + type].append(0)
                        record_based_out['noa5_' + type].append(0)
                        record_based_out['se_' + type].append(0)
                        if 'light' in events:
                            record_based_out['light_' + type].append(0)
                            record_based_out['light_fraction_' + type].append(0)
                        if 'deep' in events:
                            record_based_out['deep_' + type].append(0)
                            record_based_out['deep_fraction_' + type].append(0)
                        if 'rem' in events:
                            record_based_out['rem_' + type].append(0)
                            record_based_out['rem_fraction_' + type].append(0)

        return performance, performance_multiclass, output, record_based_out  # , analysis_output


def flatten_output_all_by_event(output, events, list_name='tar', apply_mask=True):
    out = []
    for event in events:
        event_list = []
        for rec, lists in output.items():
            new_list = lists[event][list_name]
            if apply_mask:
                new_list = [l for m, l in zip(lists['mask']['tar'], new_list) if not m]
            event_list += new_list
        out.append(event_list)
    return out

def get_biometrics(record, metric):

    # metrics can be age, bmi or sex.
    bio_path = {'arc': 'E:\\datasets\\Amazfit\\biometrics\\',
                'stages': 'E:\\datasets\\Amazfit\\biometrics\\',
                'tbi': 'E:\\datasets\\Jamie\\biometrics\\',
                'mesa': 'E:\\datasets\\mesa\\biometrics\\'}

    r = record[:-3]
    if record[:4] == 'STNF':
        df = pd.read_excel(os.path.join(bio_path['arc'], 'biometrics.xlsx'))
    elif record[:4] == 'mesa':
        df = pd.read_excel(os.path.join(bio_path['mesa'], 'biometrics.xlsx'))
        # do some processing
    else:
        df = pd.read_excel(os.path.join(bio_path['tbi'], 'biometrics.xlsx'))
        r_ = r.split('-')
        r = '{}{}'.format(r_[0], int(r_[1]))
    k = 1
    val = df[metric][df['ID'].isin([r])].values
    if len(val) == 0:
        return np.NaN
    else:
        return val[0]