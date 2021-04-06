import numpy as np
from scipy.signal.windows import hann

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class RMSFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=300, shift=15, percentile=0.1):
        self.window_size = window_size
        self.shift = shift
        self.percentile = percentile

    def _get_basal_tones(self, signal):
        basal_tones = []
        for i in range(0, len(signal) - self.window_size + 1, self.shift):
            window = signal[i:i+self.window_size]
            _min, _max = np.min(window), np.max(window)
            base_value = 0.25 * (_max - _min)
            # Take the mean of the 10% smallest values
            mean_smallest_values = np.mean(sorted(window)[:int(self.percentile * len(window))])
            
            basal_tones.append(mean_smallest_values + base_value)

        basal_tones = np.array(basal_tones)
        basal_tones = np.interp(list(range(len(signal))),
                                list(range(0, len(signal) - self.window_size + 1, self.shift)),
                                basal_tones)
        return basal_tones


    def _rms(self, signal):
        window = hann(self.window_size)
        rms_signal = []
        for i in range(0, len(signal) - self.window_size + 1, self.shift):
            subsignal = signal[i:i+self.window_size] * window
            rms_signal.append(np.sqrt(np.mean(subsignal ** 2)))
        rms_signal = np.array(rms_signal)
        rms_signal = np.interp(list(range(len(signal))),
                               list(range(0, len(signal) - self.window_size + 1, self.shift)),
                               rms_signal)
        return rms_signal

    def fit(self, X, y):
        self.names_ = [
            'RMS_mean',
            'RMS_std',
            'RMS_max',
            'RMS_min',
            'RMS_nr_zeros'
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, ['names_'])
        features = np.zeros((X.shape[0], len(self.names_)))
        for i in range(X.shape[0]):
            signal = X[i, :]
            rms_signal = self._rms(signal)
            basal_signal = self._get_basal_tones(rms_signal)
            length = min(len(rms_signal), len(basal_signal))
            diff_signal = rms_signal[:length] - basal_signal[:length]
            diff_signal[diff_signal < 0] = 0
            features[i, :] =[np.mean(diff_signal), np.std(diff_signal), 
                             np.max(diff_signal), np.min(diff_signal), 
                             np.sum(diff_signal == 0)]
        return features

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

