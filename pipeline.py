import warnings; warnings.filterwarnings('ignore')

from util import *
from features import *
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

    # Now cut up all train signals into windows for evaluation
    # train_eval_windows, train_eval_idx, all_train_eval_files = [], [], []
    # for file in tqdm(train_files, desc='Extracting train eval windows...'):
    #     windows, idx = extract_test_windows('{}/{}'.format(DATA_DIR, file), 
    #                                            window_size=WINDOW_SIZE,
    #                                            shift=WINDOW_SHIFT,
    #                                            LOW_FREQ=LOW_FREQ,
    #                                            HIGH_FREQ=HIGH_FREQ)
    #     train_eval_windows.extend(windows)
    #     train_eval_idx.extend(idx)
    #     all_train_eval_files.extend([file] * len(idx))

    # Also cut up the test signals into windows
    test_windows, test_labels, test_idx, all_test_files = [], [], [], []
    for file in tqdm(test_files, desc='Extracting test windows...'):
        windows, idx = extract_test_windows('{}/{}'.format(DATA_DIR, file),
                                            window_size=WINDOW_SIZE,
                                            shift=WINDOW_SHIFT,
                                            LOW_FREQ=LOW_FREQ,
                                            HIGH_FREQ=HIGH_FREQ)
        test_windows.extend(windows)
        test_idx.extend(idx)
        all_test_files.extend([file] * len(idx))

    # Now extract features from these windows
    X_train, X_test, ts_features, clin_features = extract_all_features(
        train_windows, train_labels, train_idx, all_train_files, 
        test_windows, test_idx, all_test_files
    )

    train_labels = pd.Series(train_labels, index=X_train.index)

    # Apply feature selection
    X_train, X_test = select_features(X_train, train_labels, X_test, 
                                      ts_features, P_VALUE=P_VALUE)

    # Create a gradient boosting model
    model = fit_model(X_train, train_labels, OUTPUT_DIR)

    # Generate the predictions for training and testing files
    for file in np.unique(all_test_files):
        generate_predictions(file, X_test.loc[X_test['file'] == file, :], np.array(test_idx)[np.array(all_test_files) == file], model, WINDOW_SIZE, DATA_DIR, OUTPUT_DIR+'/test')
    #for file in np.unique(all_train_eval_files):
    #    generate_predictions(file, X_train_eval.loc[X_train_eval['file'] == file, :], np.array(train_eval_idx)[np.array(all_train_eval_files) == file], model, WINDOW_SIZE, DATA_DIR, OUTPUT_DIR+'/train')

    # Generate predictions for the icelandic dataset
    if TRANSFER:
        pass

    # Evaluate predictions for this fold
    for file in np.unique(all_test_files):
        auc, ious = evaluate(DATA_DIR + '/' + file, pd.read_csv(OUTPUT_DIR+'/test/{}.csv'.format(file), header=None)[1].values)
        print(file)
        print('AUC = {}'.format(auc))
        print('IoUs = {}'.format(ious))
