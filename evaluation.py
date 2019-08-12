import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score

from util import *

def calc_iou(real_segment, predicted_segment):
    real_segment = set(range(real_segment[0], real_segment[1]))
    predicted_segment = set(range(predicted_segment[0], predicted_segment[1]))
    nomin = len(real_segment.intersection(predicted_segment))
    denomin = len(real_segment.union(predicted_segment))
    
    return nomin / denomin


def find_intervals(predictions, thresh=0.5):
    intervals = []
    in_interval = False
    for i, x in enumerate(predictions):
        if not in_interval and x >= thresh:
            start = i
            in_interval = True
        elif in_interval and x < thresh:
            intervals.append((start, i))
            in_interval = False

    if in_interval:
        intervals.append((start, i))
            
    return intervals


def average_iou(real_segments, predicted_segments):
    ious = []
    for real_segment in real_segments:
        for predicted_segment in predicted_segments:
            iou = calc_iou(real_segment, predicted_segment)
            if iou > 0:
                ious.append(iou)

    return np.mean(ious), np.std(ious)


def calculate_auc(intervals, predictions):
    preds = []
    labels = []
    for (start_idx, start_type), (end_idx, end_type) in zip(intervals[::2], intervals[1::2]):
        if start_type[-1] == 'C':
            labels.extend([1]*(end_idx - start_idx))
            preds.extend(predictions[start_idx:end_idx])
        else:
            labels.extend([0]*(end_idx - start_idx))
            preds.extend(predictions[start_idx:end_idx])

    return roc_auc_score(labels, preds)


def create_plot(signal_ch1, signal_ch2, signal_ch3, predictions, intervals, OUTPUT_PATH):
    f, ax = plt.subplots(4, 1, sharex=True, figsize=(15,3))
    ax[0].plot(signal_ch1)
    ax[1].plot(signal_ch2)
    ax[2].plot(signal_ch3)

    _max = np.max([np.max(signal_ch1), np.max(signal_ch2), np.max(signal_ch3)])
    _min = np.min([np.min(signal_ch1), np.min(signal_ch2), np.min(signal_ch3)])

    for (start_idx, start_type), (end_idx, end_type) in zip(intervals[::2], intervals[1::2]):
        if start_type[-1] == 'C':
            color = 'g'
        elif start_type == '(c)':
            color = 'y'
        else:
            color = 'r'

        for k in range(3):
            rect = patches.Rectangle((start_idx, _min), end_idx - start_idx, _max - _min, facecolor=color, alpha=0.5)
            ax[k].add_patch(rect)

    ax[3].plot(predictions)
    plt.savefig(OUTPUT_PATH)
    plt.close()


def generate_predictions(file, X, idx, model, WINDOW_SIZE, DATA_DIR, OUTPUT_DIR):
    for col in ['ID', 'file']:
        if col in X.columns:
            X = X.drop(col, axis=1)

    signal_ch1, signal_ch2, signal_ch3, annotated_intervals = read_signal(DATA_DIR + '/' + file)
    ts_predictions = np.empty((len(signal_ch1),), dtype=object)
    predictions = model.predict_proba(X)[:, 1]
    for pred, x in zip(predictions, idx):
      for i in range(x, x+WINDOW_SIZE):
        if ts_predictions[i] is None:
          ts_predictions[i] = [pred]
        else:
          ts_predictions[i].append(pred)
    
    for i in range(len(signal_ch1)):
      if ts_predictions[i] is None:
        ts_predictions[i] = last_value
      else:
        avg = np.mean(ts_predictions[i])
        ts_predictions[i] = avg
        last_value = avg

    pd.Series(ts_predictions).to_csv('{}/{}.csv'.format(OUTPUT_DIR, file))
    create_plot(signal_ch1, signal_ch2, signal_ch3, ts_predictions, annotated_intervals, '{}/{}.png'.format(OUTPUT_DIR, file))

def evaluate(file, predictions):
    signal_ch1, signal_ch2, signal_ch3, annotated_intervals = read_signal(file)
    auc = calculate_auc(annotated_intervals, predictions)

    real_segments = []
    for (start_idx, start_type), (end_idx, end_type) in zip(annotated_intervals[::2], annotated_intervals[1::2]):
        if start_type[-1] == 'C':
            real_segments.append((start_idx, end_idx))

    ious = []
    for thresh in np.arange(0.05, 1.0, 0.05):
        pred_segments = find_intervals(predictions, thresh=thresh)
        ious.append(average_iou(real_segments, pred_segments))

    return auc, ious


def evaluate_all(files, prediction_path, OUTPUT_DIR):
    preds = []
    labels = []
    ious = []

    for i, _ in enumerate(np.arange(0.05, 1.0, 0.05)):
        ious[i] = []

    for file in files:
        signal_ch1, signal_ch2, signal_ch3, annotated_intervals = read_signal(file)
        predictions = pd.read_csv(prediction_path+'/{}.csv'.format(file), header=None)[1].values
        for (start_idx, start_type), (end_idx, end_type) in zip(intervals[::2], intervals[1::2]):
            if start_type[-1] == 'C':
                labels.extend([1]*(end_idx - start_idx))
                preds.extend(predictions[start_idx:end_idx])
            else:
                labels.extend([0]*(end_idx - start_idx))
                preds.extend(predictions[start_idx:end_idx])

        for i, thresh in enumerate(np.arange(0.05, 1.0, 0.05)):
            pred_segments = find_intervals(predictions, thresh=thresh)
            ious[i].append(average_iou(real_segments, pred_segments))

    print('AUC = {}'.format(roc_auc_score(labels, preds)))

    iou_x = []
    iou_y = []
    iou_e = []
    for i, thresh in enumerate(np.arange(0.05, 1.0, 0.05)):
      mean_iou, std_iou = np.mean(ious[i]), np.std(ious[i])
      iou_x.append(thresh)
      iou_y.append(mean_iou)
      iou_e.append(std_iou)

    plt.figure()
    plt.errorbar(iou_x, iou_y, yerr=iou_e)
    plt.savefig(OUTPUT_DIR + '/ious.png')
    plt.close()
