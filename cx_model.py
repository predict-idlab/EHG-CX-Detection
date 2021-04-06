import numpy as np
import pandas as pd

<<<<<<< HEAD
from util_functions import extract_windows

=======
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
from collections import Counter

from catboost import CatBoostClassifier

<<<<<<< HEAD
from features import (BasicFrequencyFeatures, BasicFeatures, BOSSFeatures, 
                      CorrFeatures)
=======
from tsfresh.feature_selection.relevance import calculate_relevance_table

from util import extract_windows
from features import (BasicFrequencyFeatures, BasicFeatures, BOSSFeatures, 
                      CorrFeatures, RMSFeatures, TSFRESHFeatures, GENDISFeatures)
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84

class WindowData():
    def __init__(self):
        self.windows = []
        self.labels = []
        self.files = []
        self.indices = []


class CXDetector: # "sklearn estimator"
    def __init__(self, sample_freq, low_freq, high_freq, window_size, 
                 window_shift, shap_window_size, shap_window_shift, 
                 read_signal_fn, read_clin_fn, proc_clin_fn):

        self.sample_freq = sample_freq
        self.low_freq = low_freq
        self.high_freq = high_freq

        self.window_size = window_size
        self.window_shift = window_shift
        self.shap_window_size = shap_window_size
        self.shap_window_shift = shap_window_shift

        self.read_signal_fn = read_signal_fn
        self.read_clin_fn = read_clin_fn
        self.proc_clin_fn = proc_clin_fn

<<<<<<< HEAD
        # Features to apply on the shorter noramlized windows
        self.short_features = [

=======
        # Features to apply on the shorter normalized windows
        self.short_features = [
            GENDISFeatures,
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        ]

        # Features using 1 channel to apply on the normal windows
        self.features = [
<<<<<<< HEAD
            BOSSFeatures,
            BasicFrequencyFeatures,
            BasicFeatures
=======
            # Spectral features
            BOSSFeatures,
            BasicFrequencyFeatures,

            # Temporal features
            BasicFeatures,
            RMSFeatures,
            #TSFRESHFeatures,
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        ]

        # Features using multiple channels
        self.multi_channel_features = [
<<<<<<< HEAD
=======
            # Correlations
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
            CorrFeatures()
        ]

    def prep_data(self, files, train=True):
        # Read in our data
        signals = []
        intervals = []
        for file in files:
            *_signals, _intervals = self.read_signal_fn(file, self.low_freq, 
                                                        self.high_freq, 
                                                        self.sample_freq)
            signals.append(_signals)
            intervals.append(_intervals)

        # Extract windows from the data
        window_data = WindowData()
        shap_window_data = WindowData()
        for file, signals, intervals in zip(files, signals, intervals):
            if train:
                for ann1, ann2 in zip(intervals[::2], intervals[1::2]):

                    if (ann1[1][-1] not in ['C', 'D'] or 
                            ann2[0] >= len(signals[0]) or 
                            ann1[0] < 0):
                        continue

                    label = int(ann1[1][-1] == 'C')
                    _windows, idx = extract_windows(signals, ann1[0], ann2[0], 
                                                    self.window_size, 
                                                    self.window_shift)

                    window_data.windows.extend(_windows)
                    window_data.indices.extend(idx)
                    window_data.files.extend([file]*len(_windows))
                    window_data.labels.extend([label]*len(_windows))

                    shap_windows, shap_idx = extract_windows(signals, ann1[0], 
                                                             ann2[0], 
                                                             self.shap_window_size, 
                                                             self.shap_window_shift)

                    shap_window_data.windows.extend(shap_windows)
                    shap_window_data.indices.extend(shap_idx)
                    shap_window_data.files.extend([file]*len(shap_windows))
                    shap_window_data.labels.extend([label]*len(shap_windows))
            else:
                _windows, idx = extract_windows(signals, 0, len(signals[0]), 
                                                self.window_size, 
                                                self.window_shift)

                window_data.windows.extend(_windows)
                window_data.indices.extend(idx)
                window_data.files.extend([file]*len(_windows))
                window_data.labels.extend([None]*len(_windows))

                shap_windows, shap_idx = extract_windows(signals, 0, 
                                                         len(signals[0]), 
                                                         self.shap_window_size, 
                                                         self.shap_window_shift)

                shap_window_data.windows.extend(shap_windows)
                shap_window_data.indices.extend(shap_idx)
                shap_window_data.files.extend([file]*len(shap_windows))
                shap_window_data.labels.extend([None]*len(shap_windows))

        window_data.windows = np.array(window_data.windows)
        window_data.files = np.array(window_data.files)
        window_data.indices = np.array(window_data.indices)
        shap_window_data.windows = np.array(shap_window_data.windows)
        shap_window_data.files = np.array(shap_window_data.files)
        shap_window_data.indices = np.array(shap_window_data.indices)

        return window_data, shap_window_data

<<<<<<< HEAD
=======
    def get_corr_features(self, X):
        """Get all coordinates in the X-matrix with correlation value equals 1
        (columns with equal values), excluding elements on the diagonal.

        Parameters:
        -----------
        - train_df: pd.DataFrame
            the feature matrix where correlated features need to be removed

        Returns
        -------
        - correlated_feature_pairs: list of tuples
            coordinates (row, col) where correlated features can be found
        """
        row_idx, col_idx = np.where(np.abs(X.corr()) > 0.99)
        self_corr = set([(i, i) for i in range(X.shape[1])])
        correlated_feature_pairs = set(list(zip(row_idx, col_idx))) - self_corr
        return correlated_feature_pairs


    def get_uncorr_features(self, data):
        """Remove clusters of these correlated features, until only one feature 
        per cluster remains.

        Parameters:
        -----------
        - data: pd.DataFrame
            the feature matrix where correlated features need to be removed

        Returns
        -------
        - data_uncorr_cols: list of string
            the column names that are completely uncorrelated to eachother
        """
        X_train_corr = data.copy()
        correlated_features = self.get_corr_features(X_train_corr)

        corr_cols = set()
        for row_idx, col_idx in correlated_features:
            corr_cols.add(row_idx)
            corr_cols.add(col_idx)

        uncorr_cols = list(set(X_train_corr.columns) - set(X_train_corr.columns[list(corr_cols)]))
       
        col_mask = [False]*X_train_corr.shape[1]
        for col in corr_cols:
            col_mask[col] = True
        X_train_corr = X_train_corr.loc[:, col_mask]
      
        correlated_features = self.get_corr_features(X_train_corr)
        to_remove = set()
        for corr_row, corr_col in correlated_features:
            if corr_row in to_remove:
                continue

            for corr_row2, corr_col2 in correlated_features:
                if corr_row == corr_row2:
                    to_remove.add(corr_col2)
                elif corr_row == corr_col2:
                    to_remove.add(corr_row2)

        col_mask = [True]*X_train_corr.shape[1]
        for ix in to_remove:
            col_mask[ix] = False

        X_train_corr = X_train_corr.loc[:, col_mask]

        data_uncorr_cols = list(set(list(X_train_corr.columns) + uncorr_cols))

        return data_uncorr_cols

    def remove_features(self, data):
        """Remove all correlated features and columns with only a single value.

        Parameters:
        -----------
        - data: pd.DataFrame
            the feature matrix where correlated features need to be removed

        Returns
        -------
        - useless_cols: list of string
            list of column names that have no predictive value
        """
        single_cols = list(data.columns[data.nunique() == 1])

        uncorr_cols = self.get_uncorr_features(data)
        corr_cols = list(set(data.columns) - set(uncorr_cols))

        useless_cols = list(set(single_cols + corr_cols))

        print('Removing {} features'.format(len(useless_cols)))

        return useless_cols

>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
    def fit(self, train_files):
        window_data, shap_window_data = self.prep_data(train_files)

        # Extract clinical variables
        clin_features = []
        for file in train_files:
            names, values = self.read_clin_fn(file)
            clin_features.append([file]+values)
        clin_df = pd.DataFrame(clin_features, columns=['file']+names)
        clin_df = self.proc_clin_fn(clin_df)

        # Extract features for each channel separately
        features_per_channel = []
        self.feature_extractors_per_channel = {}
        for ch in range(window_data.windows.shape[1]):
            self.feature_extractors_per_channel[ch] = []
            for feature_extractor in self.features:
                self.feature_extractors_per_channel[ch].append(feature_extractor())

            channel_features = []
            for f in self.feature_extractors_per_channel[ch]:
<<<<<<< HEAD
                features = f.fit_transform(window_data.windows[:, ch, :], None)
=======
                features = f.fit_transform(window_data.windows[:, ch, :], window_data.labels)
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
                features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
                channel_features.append(features)
            features_per_channel.append(pd.concat(channel_features, axis=1))

<<<<<<< HEAD
        # Extract features using all channels
        features_multi_channel = []
        for f in self.multi_channel_features:
            features = f.fit_transform(window_data.windows, None)
=======
        short_features_per_channel = []
        self.short_feature_extractors_per_channel = {}
        for ch in range(window_data.windows.shape[1]):
            self.short_feature_extractors_per_channel[ch] = []
            for feature_extractor in self.short_features:
                self.short_feature_extractors_per_channel[ch].append(feature_extractor())

            channel_features = []
            for f in self.short_feature_extractors_per_channel[ch]:
                f.fit(shap_window_data.windows[:, ch, :], shap_window_data.labels)
                features = f.transform(window_data.windows[:, ch, :], window_data.labels)
                features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
                channel_features.append(features)
            short_features_per_channel.append(pd.concat(channel_features, axis=1))

        features_multi_channel = []
        for f in self.multi_channel_features:
            features = f.fit_transform(window_data.windows, window_data.labels)
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
            features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
            features_multi_channel.append(features)

        # Concatenate the features of different channels together
<<<<<<< HEAD
        train_features = pd.concat(features_per_channel + features_multi_channel, axis=1)
=======
        train_features = pd.concat(features_per_channel + short_features_per_channel + features_multi_channel, axis=1)
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        train_features['file'] = window_data.files
        train_features = train_features.merge(clin_df, on='file')

        # Create our X and y
        X_train = train_features
        y_train = np.array(window_data.labels)
        for col in ['ID', 'file']:
            if col in X_train.columns:
                X_train = X_train.drop(col, axis=1)

<<<<<<< HEAD
=======
        X_train = X_train.astype(float)

        # useless_features = self.remove_features(X_train)
        # X_train = X_train.drop(useless_features, axis=1)

        # Now apply hypothesis testing on remaining features
        rel_table = calculate_relevance_table(X_train, pd.Series(y_train))
        self.rel_features = list(rel_table[rel_table['p_value'] <= 0.05].index)

        X_train = X_train[self.rel_features]

>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        # Create validation set for early stopping
        val_files = np.random.choice(train_files, size=int(0.1 * len(train_files)), replace=False)
        all_files = np.array(window_data.files)
        X_val = X_train.loc[np.isin(window_data.files, val_files), :]
        y_val = y_train[np.isin(window_data.files, val_files)]
        X_train = X_train.loc[~np.isin(window_data.files, val_files), :]
        y_train = y_train[~np.isin(window_data.files, val_files)]

        # Fit our gradient boosting classifier
        self.clf = CatBoostClassifier(iterations=10000, od_type='Iter', od_wait=50, 
                                      objective='CrossEntropy', random_seed=2018,
                                      #eval_metric='AUC', 
                                      use_best_model=True, 
                                      task_type='CPU')
            
        self.clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

        return train_features

    def predict(self, test_files):
<<<<<<< HEAD
=======
        # TODO: Take means of all predictions on same timepoint
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        window_data, shap_window_data = self.prep_data(test_files, train=False)

        # Extract clinical variables
        clin_features = []
        for file in test_files:
            names, values = self.read_clin_fn(file)
            clin_features.append([file]+values)
        clin_df = pd.DataFrame(clin_features, columns=['file']+names)
        clin_df = self.proc_clin_fn(clin_df)

        # Extract features for each channel separately
        features_per_channel = []
        for ch in range(window_data.windows.shape[1]):
            channel_features = []
            for f in self.feature_extractors_per_channel[ch]:
                features = f.transform(window_data.windows[:, ch, :])
                features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
                channel_features.append(features)
            features_per_channel.append(pd.concat(channel_features, axis=1))

<<<<<<< HEAD
=======
        short_features_per_channel = []
        for ch in range(window_data.windows.shape[1]):
            channel_features = []
            for f in self.short_feature_extractors_per_channel[ch]:
                features = f.transform(window_data.windows[:, ch, :])
                features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
                channel_features.append(features)
            short_features_per_channel.append(pd.concat(channel_features, axis=1))

>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        features_multi_channel = []
        for f in self.multi_channel_features:
            features = f.transform(window_data.windows)
            features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
            features_multi_channel.append(features)

        # Concatenate the features of different channels together
<<<<<<< HEAD
        test_features = pd.concat(features_per_channel + features_multi_channel, axis=1)
=======
        test_features = pd.concat(features_per_channel + short_features_per_channel + features_multi_channel, axis=1)
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
        test_features['file'] = window_data.files
        test_features = test_features.merge(clin_df, on='file')

        all_preds = []
        for file in test_files:
            X_test = test_features[test_features['file'] == file]
            test_ix = window_data.indices[window_data.files == file].flatten()
            for col in ['ID', 'file']:
                if col in X_test.columns:
                    X_test = X_test.drop(col, axis=1)
<<<<<<< HEAD
=======

            X_test = X_test[self.rel_features]
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
            preds = self.clf.predict_proba(X_test)[:, 1]#.reshape(-1, 1)

            pred_df = pd.DataFrame(list(range(max(test_ix) + self.window_size)), columns=['index'])
            pred_df['file'] = file
            pred_df['pred'] = np.NaN
            pred_df = pred_df.set_index('index', drop=True)
            pred_df.loc[test_ix, 'pred'] = preds
            pred_df = pred_df.ffill().reset_index()
            all_preds.append(pred_df)

<<<<<<< HEAD
        return pd.concat(all_preds).reset_index(drop=True)
=======
        return pd.concat(all_preds).reset_index(drop=True)

"""
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

"""
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
