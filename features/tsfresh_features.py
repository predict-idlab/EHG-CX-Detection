import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

# TODO: Make a dict from EfficientFCParameters with faster features
extraction_settings = EfficientFCParameters()
filtered_funcs = [
    'abs_energy', 'mean_abs_change', 'mean_change', 'skewness', 
    'kurtosis', 'absolute_sum_of_changes', 'longest_strike_below_mean', 
    'longest_strike_above_mean', 'count_above_mean', 'count_below_mean', 
    'last_location_of_maximum', 'first_location_of_maximum', 
    'last_location_of_minimum', 'first_location_of_minimum', 
    'percentage_of_reoccurring_datapoints_to_all_datapoints', 
    'percentage_of_reoccurring_values_to_all_values', 
    'sum_of_reoccurring_values', 'sum_of_reoccurring_data_points', 
    'ratio_value_number_to_time_series_length', 'cid_ce', 
    'symmetry_looking', 'large_standard_deviation', 'quantile', 
    'autocorrelation', 'number_peaks', 'binned_entropy', 
    'index_mass_quantile', 'linear_trend',  'number_crossing_m', 
    'augmented_dickey_fuller', 'number_cwt_peaks', 'agg_autocorrelation', 
    'spkt_welch_density', 'friedrich_coefficients', 
    'max_langevin_fixed_point', 'c3', 'ar_coefficient', 
    'mean_second_derivative_central', 'ratio_beyond_r_sigma', 
    'energy_ratio_by_chunks', 'partial_autocorrelation', 'fft_aggregated', 
    'time_reversal_asymmetry_statistic', 'range_count'
]
filtered_settings = {}
for func in filtered_funcs:
  filtered_settings[func] = extraction_settings[func]

class TSFRESHFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, p_val=0.05, params=filtered_settings):
        self.p_val = p_val
        self.params = params
        self.fitted = False

    def fit(self, X, y):
        self.y = pd.Series(y)
        return self

    def transform(self, X):
        # Change X to a dataframe
        dfs = []
        for i in range(X.shape[0]):
            df = pd.DataFrame(X[i, :].T, columns=['value'])
            df['id'] = i
            dfs.append(df)
        tsfresh_df = pd.concat(dfs)
        
        # Extract features with tsfresh
        tsfresh_features = extract_features(tsfresh_df, impute_function=impute, 
                                            column_id='id', chunksize=None,
                                            default_fc_parameters=self.params,
                                            show_warnings=False, n_jobs=2,
                                            disable_progressbar=True)

        # Apply feature selection if we are in fitting phase
        # TODO: Move this feature selection to the cx_model file on all features
        if not self.fitted:
            self.names_ = list(tsfresh_features.columns)
            self.fitted = True
            pass

        return tsfresh_features.values

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)