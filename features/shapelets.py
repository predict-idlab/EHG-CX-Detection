import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
<<<<<<< HEAD
from genetic import GeneticExtractor
from pairwise_dist import _pdist, _pdist_location
=======

from gendis.genetic import GeneticExtractor
from gendis.pairwise_dist import _pdist, _pdist_location
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def auc_fitness_location(X, y, shapelets, cache=None, verbose=False):
    """Calculate the fitness of an individual/shapelet set"""
    D = np.zeros((len(X), len(shapelets)))
    L = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances & locations for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        
        cache_val = cache.get(('dist', shap_hash))
        if cache_val is not None:
            D[:, shap_ix] = cache_val
            
        cache_val = cache.get(('loc', shap_hash))
        if cache_val is not None:
            L[:, shap_ix] = cache_val

    # Fill up the 0 entries
    _pdist_location(X, [shap.flatten() for shap in shapelets], D, L)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(('dist', shap_hash), D[:, shap_ix])
        cache.set(('loc', shap_hash), L[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(np.hstack((D, L)), y)
    preds = lr.predict_proba(np.hstack((D, L)))
    cv_score = abs(roc_auc_score(y, preds[:, 1]) - 0.5)

    return (cv_score, sum([len(x) for x in shapelets]))

class GENDISFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

<<<<<<< HEAD
=======
    def fit(self, X, y):
        print(X.shape)
        self.genetic_extractor = GeneticExtractor(verbose=True, population_size=25, 
                                     iterations=10, wait=5, max_len=50,
                                     plot=None, location=True, n_jobs=4,
                                     fitness=auc_fitness_location)
        self.genetic_extractor.fit(X, y)
        self.names = []
        for i, shap in enumerate(self.genetic_extractor.shapelets):
            self.names.append('dist_shap_{}'.format(i))
        for i, shap in enumerate(self.genetic_extractor.shapelets):
            self.names.append('loc_shap_{}'.format(i))

        return self

    def transform(self, X):
        return self.genetic_extractor.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84
