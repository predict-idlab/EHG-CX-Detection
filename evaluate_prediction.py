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

for directory in os.listdir(RESULTS_DIR):
    print(directory)
    if 'config.json' in os.listdir(RESULTS_DIR + '/' + directory):
    	print(json.loads(open(RESULTS_DIR + '/' + directory + '/config.json', 'r').read()))
    print('Unweighted AUC = {}'.format(unweighted_auc(RESULTS_DIR + '/' + directory)))
    print('CX Weighted AUC = {}'.format(cx_weighted_auc(RESULTS_DIR + '/' + directory)))
    print('Signal + CX Weighted AUC = {}'.format(signal_cx_weighted_auc(RESULTS_DIR + '/' + directory)))