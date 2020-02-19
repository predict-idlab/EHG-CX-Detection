import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from nolitsa.lyapunov import mle_embed
from nolitsa.d2 import c2_embed, d2
import statsmodels.api as sm
from sklearn.neighbors import KDTree
from entropy.utils import _embed

class BasicFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, r=0.15, sampen_order=2, Q=7, lyap_maxt=10, metric='chebyshev'):
        self.r = r
        self.sampen_order = sampen_order
        self.Q = Q
        self.lyap_maxt = lyap_maxt
        self.metric = metric

    def _log2(self, data):
        return np.exp(np.mean(np.log(np.abs(data))))

    def _time_reversibility(self, data):
        norm = 1 / (len(data) - 1)
        lagged_data = data[1:]
        return norm * np.sum(np.power((lagged_data - data[:-1]), 3))

    def _ac_zero_crossing(self, signal):
        tau = sm.tsa.acf(signal, nlags=len(signal) - 1)
        tau_neg_ix = np.arange(len(tau), dtype=int)[tau < 0]
        return tau_neg_ix[0]

    def _max_lyap(self, signal, ac_zero):
        y = mle_embed(signal, [self.Q], ac_zero, maxt=self.lyap_maxt)[0]
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def _corr_dim(self, signal, ac_zero):
        r, C_r = c2_embed(signal, [self.Q], ac_zero)[0]
        return d2(r[:self.Q], C_r[:self.Q])[0]

    def _simple_square_integral(self, data):
        return np.sum(np.power(data, 2))

    def _sampen(self, x):
        """Utility function for `app_entropy`` and `sample_entropy`.
        FROM: https://github.com/raphaelvallat/entropy/blob/master/entropy/entropy.py
        """
        phi = np.zeros(2)
        if self.r is None:
            r = 0.2 * np.std(x, axis=-1, ddof=1)
        else:
            r = self.r * np.std(x, axis=-1, ddof=1)

        # compute phi(order, r)
        _emb_data1 = _embed(x, self.sampen_order, 1)
        emb_data1 = _emb_data1
        count1 = KDTree(emb_data1, metric=self.metric).query_radius(emb_data1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(order + 1, r)
        emb_data2 = _embed(x, self.sampen_order + 1, 1)
        count2 = KDTree(emb_data2, metric=self.metric).query_radius(emb_data2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))

        return np.subtract(phi[0], phi[1])

    def fit(self, X, y):
        self.names_ = [
            'log2',
            'time_reversibility',
            #'ac_zero_crossing',
            #'max_lyapunov',
            #'corr_dim',
            'square_integral',
            'sampen'
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, ['names_'])
        features = np.zeros((X.shape[0], len(self.names_)))
        for i in range(X.shape[0]):
                log2 = self._log2(X[i, :])
                time_rev = self._time_reversibility(X[i, :])
                #ac_zero = self._ac_zero_crossing(X[i, :])
                #max_lyap = self._max_lyap(X[i, :], ac_zero)
                #corr_dim = self._corr_dim(X[i, :], ac_zero)
                sq_int = self._simple_square_integral(X[i, :])
                sampen = self._sampen(X[i, :])
                features[i, :] = [
                    log2, time_rev, #ac_zero, max_lyap,
                    #corr_dim, 
                    sq_int, sampen
                ]
        return features

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
