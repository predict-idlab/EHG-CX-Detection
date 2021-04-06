import warnings; warnings.filterwarnings('ignore')

from util import *
from old_features import *
from evaluation import *
from model import *

from tqdm import tqdm

import numpy as np
import pandas as pd

import datetime
import os
import json

parameters = json.load(open('config.json', 'r'))

# Parameters
DATA_DIR = 'tpehgts'
WINDOW_SIZE = parameters['WINDOW_SIZE']
WINDOW_SHIFT = parameters['WINDOW_SHIFT']
P_VALUE = parameters['P_VALUE']
LOW_FREQ = parameters['LOW_FREQ']
HIGH_FREQ = parameters['HIGH_FREQ']
TRANSFER = parameters['TRANSFER']

# Output everything to a specific directory
today = datetime.datetime.now()
OUTPUT_DIR = 'output/{}_{}_{}_{}_{}'.format(today.year, today.month, 
                                            today.day, today.hour, 
                                            today.minute)
os.mkdir(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR + '/train')
os.mkdir(OUTPUT_DIR + '/test')
json.dump(parameters, open(OUTPUT_DIR+'/config.json', 'w+'))

folds = partition_data(DATA_DIR)

for train_files, test_files in folds:
    # Let's first extract windows from the annotated parts of the signal,
    # which will be used to train our model
    train_windows, train_labels, train_idx, all_train_files = [], [], [], []
    norm_train_windows, norm_train_labels, norm_train_idx, norm_all_train_files = [], [], [], []
    for file in tqdm(train_files, desc='Extracting train fit windows...'):
        windows, labels, idx = extract_train_windows('{}/{}'.format(DATA_DIR, file), 
                                                     window_size=WINDOW_SIZE,
                                                     shift=WINDOW_SHIFT,
                                                     LOW_FREQ=LOW_FREQ,
                                                     HIGH_FREQ=HIGH_FREQ)
        train_windows.extend(windows)
        train_labels.extend(labels)
        train_idx.extend(idx)
        all_train_files.extend([file] * len(labels))

        norm_windows, labels, idx = extract_train_windows('{}/{}'.format(DATA_DIR, file),
                                                          window_size=WINDOW_SIZE,
                                                          shift=WINDOW_SHIFT,
                                                          LOW_FREQ=LOW_FREQ,
                                                          HIGH_FREQ=HIGH_FREQ,
                                                          norm=True)
        norm_train_windows.extend(norm_windows)
        norm_train_labels.extend(labels)
        norm_train_idx.extend(idx)
        norm_all_train_files.extend([file] * len(labels))

    # Also cut up the test signals into windows
    test_windows, test_labels, test_idx, all_test_files = [], [], [], []
    norm_test_windows, norm_test_labels, norm_test_idx, norm_all_test_files = [], [], [], []
    for file in tqdm(test_files, desc='Extracting test windows...'):
        windows, idx = extract_test_windows('{}/{}'.format(DATA_DIR, file),
                                            window_size=WINDOW_SIZE,
                                            shift=WINDOW_SHIFT,
                                            LOW_FREQ=LOW_FREQ,
                                            HIGH_FREQ=HIGH_FREQ)
        test_windows.extend(windows)
        test_idx.extend(idx)
        all_test_files.extend([file] * len(idx))

        norm_windows, idx = extract_test_windows('{}/{}'.format(DATA_DIR, file),
                                                 window_size=WINDOW_SIZE,
                                                 shift=WINDOW_SHIFT,
                                                 LOW_FREQ=LOW_FREQ,
                                                 HIGH_FREQ=HIGH_FREQ,
                                                 norm=True)
        norm_test_windows.extend(norm_windows)
        norm_test_idx.extend(idx)
        norm_all_test_files.extend([file] * len(idx))

    # Now extract features from these windows
    X_train, X_test, ts_features, clin_features = extract_all_features(
        train_windows, train_labels, train_idx, all_train_files, 
        test_windows, test_idx, all_test_files
    )

    X_train_norm, X_test_norm, ts_features_norm, clin_features_norm = extract_all_features(
        norm_train_windows, norm_train_labels, norm_train_idx, norm_all_train_files, 
        norm_test_windows, norm_test_idx, norm_all_test_files
    )
    X_train_norm = X_train_norm.drop(clin_features_norm, axis=1)
    X_test_norm = X_test_norm.drop(clin_features_norm, axis=1)
    col_map = {}
    for feature in ts_features_norm:
        col_map[feature] = '{}_norm'.format(feature)
        ts_features.append('{}_norm'.format(feature))
    X_train_norm = X_train_norm.rename(columns=col_map)
    X_test_norm = X_test_norm.rename(columns=col_map)

    X_train = pd.concat([X_train, X_train_norm], axis=1)
    X_test = pd.concat([X_test, X_test_norm], axis=1)

    train_labels = pd.Series(train_labels, index=X_train.index)

    # Remove highly correlated features
    X = pd.concat([X_train, X_test])
    useless_features = remove_features(X)
    X_train = X_train.drop(useless_features, axis=1)
    X_test = X_test.drop(useless_features, axis=1)

    # Apply feature selection
    X_train, X_test = select_features(X_train, train_labels, X_test, 
                                      ts_features, P_VALUE=P_VALUE)

    # Create a gradient boosting model
    model = fit_model(X_train, train_labels, OUTPUT_DIR)

    # Generate the predictions for training and testing files
    for file in np.unique(all_test_files):
        generate_predictions(file, X_test.loc[X_test['file'] == file, :], np.array(test_idx)[np.array(all_test_files) == file], model, WINDOW_SIZE, DATA_DIR, OUTPUT_DIR+'/test')

    # Generate predictions for the icelandic dataset
    if TRANSFER:
        pass

    # Evaluate predictions for this fold
    for file in np.unique(all_test_files):
        auc, ious = evaluate(DATA_DIR + '/' + file, pd.read_csv(OUTPUT_DIR+'/test/{}.csv'.format(file), header=None)[1].values)
        print(file)
        print('AUC = {}'.format(auc))
        print('IoUs = {}'.format(ious))
