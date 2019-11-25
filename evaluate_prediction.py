import os
import glob
import json

from sklearn.metrics import roc_auc_score

import pandas as pd

from evaluation import *
from util import *

RESULTS_DIR = 'output'
DATA_DIR = 'tpehgts'


def _load_pred_labels_intervals(file):
    id = file.split('/')[-1].split('.')[0]
    _, _, _, intervals = read_signal(DATA_DIR + '/' + id)
    predictions = pd.read_csv(file, header=None)[1].values
    labels, preds = get_labels_preds(intervals, predictions)
    return labels, preds, intervals


# Calculate unweighted AUC by taking the predictions from the annotated windows
def unweighted_auc(directory):
    pred_files = glob.glob(directory + '/test/*.csv')
    all_labels, all_preds = [], []
    for file in pred_files:
        labels, preds, intervals = _load_pred_labels_intervals(file)
        all_labels.extend(labels)
        all_preds.extend(preds)

    return roc_auc_score(all_labels, all_preds)


# Calculate weighted AUC by taking the predictions from the annotated windows
# and giving those predictions weight 1/len(window)
def cx_weighted_auc(directory):
    pred_files = glob.glob(directory + '/test/*.csv')
    all_labels, all_preds, all_weights = [], [], []
    for file in pred_files:
        labels, preds, intervals = _load_pred_labels_intervals(file)
        all_labels.extend(labels)
        all_preds.extend(preds)

        for (start, _), (end, _) in zip(intervals[::2], intervals[1::2]):
            all_weights.extend([1 / (end - start)] * (end - start))

    return roc_auc_score(all_labels, all_preds, sample_weight=all_weights)

# Calculate weighted AUC by taking the predictions from the annotated windows
# and giving those predictions weight 1/len(window) and another normalization
# based on the number of contractions in each signal
def signal_cx_weighted_auc(directory):
    pred_files = glob.glob(directory + '/test/*.csv')
    all_labels, all_preds, all_weights = [], [], []
    for file in pred_files:
        labels, preds, intervals = _load_pred_labels_intervals(file)
        all_labels.extend(labels)
        all_preds.extend(preds)

        for (start, _), (end, _) in zip(intervals[::2], intervals[1::2]):
            all_weights.extend([1 / ((end - start) * len(intervals))] * (end - start))

    return roc_auc_score(all_labels, all_preds, sample_weight=all_weights)

score_per_config = {}
for directory in os.listdir(RESULTS_DIR):
    print(directory)
    config = json.loads(open(RESULTS_DIR + '/' + directory + '/config.json', 'r').read())
    if 'config.json' in os.listdir(RESULTS_DIR + '/' + directory):
    	print(config)
    auc_unw = unweighted_auc(RESULTS_DIR + '/' + directory)
    print('Unweighted AUC = {}'.format(auc_unw))
    auc_unw_cx = cx_weighted_auc(RESULTS_DIR + '/' + directory)
    print('CX Weighted AUC = {}'.format(auc_unw_cx))
    auc_unw_cx_signal = signal_cx_weighted_auc(RESULTS_DIR + '/' + directory)
    print('Signal + CX Weighted AUC = {}'.format(auc_unw_cx_signal))
    score_per_config[tuple(config.items())] = auc_unw + auc_unw_cx + auc_unw_cx_signal

for k, v in sorted(score_per_config.items(), key=lambda x: -x[1])[:5]:
	print(dict(k), v)