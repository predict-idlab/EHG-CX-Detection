import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class BasicFrequencyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, fs=20.0, bands=[(0.08, 1), (1, 2.2), (2.2, 3.5), (3.5, 5)]):
        self.fs = fs
        self.bands = bands

    def _fft(self, signal):
        ps = np.abs(np.fft.fft(signal))**2
        freqs = np.fft.fftfreq(len(ps), d=1/self.fs)
    
        mask = freqs >= 0
        freqs = freqs[mask]
        ps = ps[mask]

        return ps, freqs

    def _peak_freq(self, signal, low, high):
        ps, freqs = self._fft(signal)
        start = np.argmax(freqs >= low)
        end = np.argmin(freqs <= high)
        return np.abs(freqs[np.argmax(ps[start:end])])

    def _median_freq(self, signal, low, high):
        ps, freqs = self._fft(signal)
        best_k, min_dist = None, float('inf')

        start = np.argmax(freqs >= low)
        end = np.argmin(freqs <= high)
        ps = ps[start:end]

        k = len(ps) // 2
        offset = len(ps) // 4
        while offset > 0:
            sum1 = np.sum(ps[:k])
            sum2 = np.sum(ps[k:])
            d = abs(sum1 - sum2)
                
            if d < min_dist:
                min_dist = d
                best_k = k
            
            if sum1 > sum2:
                k -= offset
            else:
                k += offset
                
            offset = offset // 2
                
        return freqs[best_k]

    def fit(self, X, y):
        self.names_ = []
        for i, (low, high) in enumerate(self.bands):
            self.names_.append('peak_freq_b{}'.format(i))
            self.names_.append('median_freq_b{}'.format(i))
        return self

    def transform(self, X):
        check_is_fitted(self, ['names_'])
        features = np.zeros((X.shape[0], len(self.names_)))
        for i in range(X.shape[0]):
            for b, (low, high) in enumerate(self.bands):
                features[i, b*2] = self._peak_freq(X[i, :], low, high)
                features[i, b*2 + 1] = self._median_freq(X[i, :], low, high)

        return features

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)