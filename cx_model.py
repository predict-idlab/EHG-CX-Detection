import numpy as np
import pandas as pd

from util import extract_windows

from collections import Counter

from catboost import CatBoostClassifier

from features import (BasicFrequencyFeatures, BasicFeatures, BOSSFeatures, 
                      CorrFeatures)

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

        # Features to apply on the shorter noramlized windows
        self.short_features = [

        ]

        # Features using 1 channel to apply on the normal windows
        self.features = [
            BOSSFeatures,
            BasicFrequencyFeatures,
            BasicFeatures
        ]

        # Features using multiple channels
        self.multi_channel_features = [
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
                window_data.labels.extend([label]*len(_windows))

                shap_windows, shap_idx = extract_windows(signals, 0, 
                                                         len(signals[0]), 
                                                         self.shap_window_size, 
                                                         self.shap_window_shift)

                shap_window_data.windows.extend(shap_windows)
                shap_window_data.indices.extend(shap_idx)
                shap_window_data.files.extend([file]*len(shap_windows))
                shap_window_data.labels.extend([label]*len(shap_windows))

        window_data.windows = np.array(window_data.windows)
        window_data.files = np.array(window_data.files)
        window_data.indices = np.array(window_data.indices)
        shap_window_data.windows = np.array(shap_window_data.windows)
        shap_window_data.files = np.array(shap_window_data.files)
        shap_window_data.indices = np.array(shap_window_data.indices)

        return window_data, shap_window_data

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
                features = f.fit_transform(window_data.windows[:, ch, :], None)
                features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
                channel_features.append(features)
            features_per_channel.append(pd.concat(channel_features, axis=1))

        features_multi_channel = []
        for f in self.multi_channel_features:
            features = f.fit_transform(window_data.windows, None)
            features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
            features_multi_channel.append(features)

        # Concatenate the features of different channels together
        train_features = pd.concat(features_per_channel + features_multi_channel, axis=1)
        train_features['file'] = window_data.files
        train_features = train_features.merge(clin_df, on='file')

        # Create our X and y
        X_train = train_features
        y_train = np.array(window_data.labels)
        for col in ['ID', 'file']:
            if col in X_train.columns:
                X_train = X_train.drop(col, axis=1)

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
        window_data, shap_window_data = self.prep_data(test_files)

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

        features_multi_channel = []
        for f in self.multi_channel_features:
            features = f.transform(window_data.windows)
            features = pd.DataFrame(features, columns=['{}_ch{}'.format(x, ch) for x in f.names_])
            features_multi_channel.append(features)

        # Concatenate the features of different channels together
        test_features = pd.concat(features_per_channel + features_multi_channel, axis=1)
        test_features['file'] = window_data.files
        test_features = test_features.merge(clin_df, on='file')

        all_preds = []
        for file in test_files:
            X_test = test_features[test_features['file'] == file]
            test_ix = window_data.indices[window_data.files == file].flatten()
            for col in ['ID', 'file']:
                if col in X_test.columns:
                    X_test = X_test.drop(col, axis=1)
            preds = self.clf.predict_proba(X_test)[:, 1]#.reshape(-1, 1)

            pred_df = pd.DataFrame(list(range(max(test_ix) + self.window_size)), columns=['index'])
            pred_df['file'] = file
            pred_df['pred'] = np.NaN
            pred_df = pred_df.set_index('index', drop=True)
            pred_df.loc[test_ix, 'pred'] = preds
            pred_df = pred_df.ffill().reset_index()
            all_preds.append(pred_df)

        return pd.concat(all_preds).reset_index(drop=True)