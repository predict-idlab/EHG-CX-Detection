
from tsfresh.feature_selection.relevance import calculate_relevance_table

from catboost import CatBoostClassifier

import shap

import numpy as np
import pandas as pd

import logging
from collections import Counter
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

logging.getLogger("tsfresh").setLevel(logging.ERROR)

def get_corr_features(X):
    row_idx, col_idx = np.where(np.abs(X.corr()) > 0.99)
    self_corr = set([(i, i) for i in range(X.shape[1])])
    correlated_feature_pairs = set(list(zip(row_idx, col_idx))) - self_corr
    return correlated_feature_pairs


def get_uncorr_features(data):
    X_train_corr = data.copy()
    correlated_features = get_corr_features(X_train_corr)

    corr_cols = set()
    for row_idx, col_idx in correlated_features:
        corr_cols.add(row_idx)
        corr_cols.add(col_idx)

    uncorr_cols = list(set(X_train_corr.columns) - set(X_train_corr.columns[list(corr_cols)]))
   
    col_mask = [False]*X_train_corr.shape[1]
    for col in corr_cols:
        col_mask[col] = True
    X_train_corr = X_train_corr.loc[:, col_mask]
  
    correlated_features = get_corr_features(X_train_corr)

    while correlated_features:
        corr_row, corr_col = correlated_features.pop()
        col_mask = [True]*X_train_corr.shape[1]
        col_mask[corr_row] = False
        X_train_corr = X_train_corr.loc[:, col_mask]
        correlated_features = get_corr_features(X_train_corr)

    data_uncorr_cols = list(set(list(X_train_corr.columns) + uncorr_cols))

    return data_uncorr_cols

def remove_features(data):
    single_cols = list(data.columns[data.nunique() == 1])

    uncorr_cols = get_uncorr_features(data)
    corr_cols = list(set(data.columns) - set(uncorr_cols))

    useless_cols = list(set(single_cols + corr_cols))

    return useless_cols


def select_features(X_train, y_train, X_train_eval, X_test, ts_features, P_VALUE=0.05):
    useless_cols = remove_features(X_train)

    train_ts_features = X_train[ts_features]
    not_na_idx = X_train[ts_features].dropna().index
    rel_table = calculate_relevance_table(train_ts_features.loc[not_na_idx], 
                                          y_train.loc[not_na_idx])

    rel_features = list(rel_table[rel_table['p_value'] <= P_VALUE].index)
    unrel_ts_features = list((set(ts_features) - set(rel_features)).union(useless_cols) - {'file'})

    print('Dropping {} features...'.format(len(unrel_ts_features)))

    X_train = X_train.drop(unrel_ts_features, axis=1)
    X_train_eval = X_train_eval.drop(unrel_ts_features, axis=1)
    X_test = X_test.drop(unrel_ts_features, axis=1)

    return X_train, X_train_eval, X_test


def fit_model(X_train, y_train, output_path):
    train_files_sampled = np.random.choice(X_train['file'].drop_duplicates(), 
                             replace=False, 
                             size=int(np.floor(0.9 * len(X_train['file'].drop_duplicates()))))
    train_idx = X_train[X_train['file'].isin(train_files_sampled)].index
    val_idx = list(set(X_train.index) - set(train_idx))

    print('Fitting model...')
    print('Feature matrix shape: {}'.format(X_train.shape))
    print('Label distribution: {}'.format(Counter(y_train)))

    for col in ['ID', 'file']:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)

    X_val = X_train.loc[val_idx, :]
    y_val = y_train.loc[val_idx]
    X_train = X_train.loc[train_idx, :]
    y_train = y_train.loc[train_idx]

    clf = CatBoostClassifier(iterations=1000, od_type='Iter', od_wait=50, 
                             objective='CrossEntropy', random_seed=2018,
                             #eval_metric='AUC',
                             use_best_model=True, task_type='CPU')
        
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    clf.save_model(output_path+'/'+'{}.model'.format(time.time()))

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    plt.figure()
    shap.summary_plot(shap_values, X_train, max_display=50, 
                      auto_size_plot=True, show=False, 
                      color_bar=False)
    plt.gcf().set_size_inches(12, 16)
    plt.subplots_adjust(left=0.5)
    plt.savefig(output_path+'/'+'shap.svg')
    plt.close()

    return clf
