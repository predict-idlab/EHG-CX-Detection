import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from pyts.transformation import BOSS

class BOSSFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.boss = BOSS(numerosity_reduction=False)
        self.boss.fit(X, y)

        self.names_ = []
        for i in range(len(self.boss.vocabulary_)):
            self.names_.append('boss_{}'.format(self.boss.vocabulary_[i]))
        return self

    def transform(self, X):
        check_is_fitted(self, ['names_'])
        return self.boss.transform(X).toarray()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
