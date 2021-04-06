import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import permutations

class CorrFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, bins=15):
        self.bins = bins

    def _calc_non_linear_corr(self, X, Y):
        _, edges = np.histogram(X, bins=self.bins)
        bin_mids = [edges[0]]
        averages = [np.mean(Y[X <= edges[0]])]
        
        #print(Y[X < edges[0]])
        
        for i in range(0, len(edges) - 1):
            # Divide X into bins --> calculate their midpoints (Pi)
            bin_mids.append((edges[i] + edges[i + 1]) / 2)
            # Calculate the average of Y values in each bin (Qi)
            averages.append(np.mean(Y[(X >= edges[i]) & (X < edges[i + 1])]))

        bin_mids.append(edges[-1])
        averages.append(np.mean(Y[X >= edges[-1]]))
            
        bin_mids = np.array(bin_mids)
        averages = np.array(averages)
        averages = np.interp(bin_mids, bin_mids[~np.isnan(averages)], averages[~np.isnan(averages)])
            
        # Fit piecewise straight lines to (Pi, Qi)
        piecewise_fit = np.interp(X, bin_mids, averages)
            
        # Apply formula
        y_sq_sum = np.sum(np.power(Y, 2))
        
        if y_sq_sum == 0:
            return 0
        
        nominator = y_sq_sum - np.sum(np.power(Y - piecewise_fit, 2))
        return nominator / y_sq_sum

    def fit(self, X, y):
        self.names_ = []
        self.permutations = list(permutations(range(X.shape[1]), 2))
        for ch1, ch2 in self.permutations:
            self.names_.append('non_lin_corr_ch{}_ch{}'.format(ch1, ch2))

    def transform(self, X):
        check_is_fitted(self, ['names_'])
        features = np.zeros((X.shape[0], len(self.names_)))
        for i in range(X.shape[0]):
            for j, (ch1, ch2) in enumerate(self.permutations):
                features[i, j] = self._calc_non_linear_corr(X[i, ch1, :], X[i, ch2, :])
        return features


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)