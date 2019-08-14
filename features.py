import pandas as pd 
import numpy as np

from pyts.transformation import BOSS

from librosa.core import resample
from scipy.signal.windows import hann

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

import nolds

from tqdm import tqdm

import itertools
import multiprocessing
from functools import partial

n_cpu = multiprocessing.cpu_count()

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def calc_non_linear_corr(X, Y, bins=13):
    _, edges = np.histogram(X, bins=bins)
    bin_mids = [np.min(X)]
    averages = [np.min(Y)]
    for i in range(len(edges) - 1):
        # Divide X into bins --> calculate their midpoints (Pi)
        bin_mids.append((edges[i] + edges[i + 1]) / 2)
        # Calculate the average of Y values in each bin (Qi)
        y_vals = Y[(X >= edges[i]) & (X < edges[i + 1])]
        averages.append(np.mean(y_vals))
    bin_mids.append(np.max(X))
    averages.append(np.max(Y))
        
    # Fit piecewise straight lines to (Pi, Qi)
    piecewise_fit = np.interp(X, bin_mids, averages)
        
    # Apply formula
    y_sq_sum = np.sum(np.power(Y, 2))
    
    if y_sq_sum ==0:
        return 0
    
    nominator = y_sq_sum - np.sum(np.power(Y - piecewise_fit, 2))
    return nominator / y_sq_sum


def calculate_window_correlation(window):
    # Calculate pairwise non-linear correlation coefficients between all possible
    # (ordered) combinations from 2 out of #channels in the window.
    n_channels = len(window)
    coefs = []
    names = []
    for ch1, ch2 in itertools.permutations(range(n_channels), 2):
        coefs.append(calc_non_linear_corr(window[ch1], window[ch2]))
        names.append('non_lin_corr_{}_{}'.format(ch1, ch2))
        
    return coefs, names


def get_basal_tones(signal, window_size=300, shift=15, percentile=0.1, hz=20):
    basal_tones = []
    differences = []
    for i in range(0, len(signal) - window_size + 1, shift):
        window = signal[i:i+window_size]
        _min, _max = np.min(window), np.max(window)
        base_value = 0.25 * (_max - _min)
        # Take the mean of the 10% smallest values
        mean_smallest_values = np.mean(sorted(window)[:int(percentile * len(window))])
        
        basal_tones.append(mean_smallest_values + base_value)
        differences.append(2 * (base_value - mean_smallest_values))
        
    basal_tones = np.array(basal_tones)
    basal_tones = resample(basal_tones, hz / (len(signal) / len(basal_tones)), hz)
    
    differences = np.array(differences)
    differences = resample(differences, hz / (len(signal) / len(differences)), hz)
    
    return basal_tones, differences


def rms(signal, window_size=300, shift=15, hz=20):
    window = hann(window_size)
    rms_signal = []
    for i in range(0, len(signal) - window_size + 1, shift):
        subsignal = signal[i:i+window_size] * window
        rms_signal.append(np.sqrt(np.mean(subsignal ** 2)))
    rms_signal = np.array(rms_signal)
    rms_signal = resample(rms_signal, hz / (len(signal) / len(rms_signal)), hz)
    
    return rms_signal


def calculate_rms_basal_diff(window):
    # For each channel, we construct is RMS signal and basal signal
    # Then, we extract mean, min, max and std of the difference of those two signals
    n_channels = len(window)
    features = []
    names = []
    
    for ch in range(n_channels):
        signal = window[ch]
        rms_signal = rms(signal)
        basal_signal, _ = get_basal_tones(signal)
        length = min(len(rms_signal), len(basal_signal))
        diff_signal = rms_signal[:length] - basal_signal[:length]
        diff_signal[diff_signal < 0] = 0
        
        features.extend([np.mean(diff_signal), np.std(diff_signal), 
                         np.max(diff_signal), np.min(diff_signal), 
                         np.sum(diff_signal == 0)])
        names.extend(['mean_{}'.format(ch), 'std_{}'.format(ch), 'max_{}'.format(ch), 'min_{}'.format(ch), 'zeros_{}'.format(ch)])
        
    return features, names


def rowwise_chebyshev(x, y):
    return np.max(np.abs(x - y), axis=1)


def delay_embedding(data, emb_dim, lag=1):
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = "cannot embed data of length {} with embedding dimension {} " \
            + "and lag {}, minimum required length is {}"
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    return data[indices]

#C = 1.*np.array([len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))])

def sampen(data, m=3, tol=0.15, dist=rowwise_chebyshev):
    data = np.asarray(data)
    
    tol *= np.std(data)
      
    n = len(data)
    tVecs = delay_embedding(np.asarray(data), m, lag=1)
    counts = []
    
    # Calculate c_{m} and c_{m - 1}
    for m in [m - 1, m]:
        counts.append(0)
        # get the matrix that we need for the current m
        tVecsM = tVecs[:, :m]
        # successively calculate distances between each pair of template vectors
        for i in range(1, len(tVecsM)):
            dsts = dist(np.roll(tVecsM, i, axis=0), tVecsM) 
            # count how many distances are smaller than the tolerance
            counts[-1] += np.sum(dsts <= tol)
            
    if counts[1] == 0 or counts[0] == 0:
        # log would be infinite => cannot determine saen
        saen = -np.log((n - m) / (n - m - 1))
    else:
        saen = -np.log(counts[1] / counts[0])
    return saen

def calc_sampen(window, m=3, r=0.15):
    n_channels = len(window)
    features = []
    names = []
    
    for ch in range(n_channels):
        names.append('sampen_{}'.format(ch))
        features.append(sampen(window[ch], m=m, tol=r))
        
    return features, names


def median_freq(data, low, high, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    start = np.argmax(freqs >= low)
    end = np.argmin(freqs <= high)
    
    best_k, min_dist = None, float('inf')
    
    for k in range(start, end):
        d = abs(np.sum(ps[start:k]) - np.sum(ps[k:end]))
        if d < min_dist:
            min_dist = d
            best_k = k
            
    return freqs[best_k]


def peak_freq(data, low, high, fs):
    ps = np.abs(np.fft.fft(data))**2
    M = len(ps)
    freqs = np.fft.fftfreq(M, d=1/fs)
    
    start = np.argmax(freqs >= low)
    end = np.argmin(freqs <= high)
    
    return np.max(ps[start:end] / np.max(ps))


def log2(data):
    return np.exp(np.mean(np.log(np.abs(data))))


def time_reversibility(data):
    norm = 1 / (len(data) - 1)
    lagged_data = data[1:]
    return norm * np.sum(np.power((lagged_data - data[:-1]), 3))


def lyapunov(data):
    return nolds.lyap_e(data)


def simple_square_integral(data):
    return np.sum(np.power(data, 2))


def extract_window_features(params, queue=None):
    window, file, idx = params
    window = np.array(window)
    df = pd.DataFrame(window.T)
    df.columns = ['EHG_0', 'EHG_1', 'EHG_2']
    df['id'] = 0
    tsfresh_features = extract_features(df, impute_function=impute, column_id='id',
                                        default_fc_parameters=EfficientFCParameters(),
                                        show_warnings=False, n_jobs=1,
                                        chunksize=1,
                                        disable_progressbar=True)
    tsfresh_feature_names = list(tsfresh_features.columns)
    tsfresh_features = list(tsfresh_features.values[0, :])
    
    corr_coefs, corr_names = calculate_window_correlation(window)
    
    rms_features, rms_names = calculate_rms_basal_diff(window)
    
    sampen_features, sampen_names = calc_sampen(window, m=3, r=0.15)
    
    med_freq_feature_names = []
    for ch in range(3):
        for band in range(4):
            med_freq_feature_names.append('median_freq_b{}_ch{}'.format(band, ch))
            
    med_freq_features = []
    for ch in range(3):
        for low, high in [(0.08, 1), (1, 2.2), (2.2, 3.5), (3.5, 5)]:
            med_freq_features.append(median_freq(window[ch], low, high, 20))
    
    peak_freq_feature_names = []
    for ch in range(3):
        for band in range(4):
            peak_freq_feature_names.append('peak_freq_b{}_ch{}'.format(band, ch))
            
    peak_freq_features = []
    for ch in range(3):
        for low, high in [(0.08, 1), (1, 2.2), (2.2, 3.5), (3.5, 5)]:
            peak_freq_features.append(peak_freq(window[ch], low, high, 20))

    log_features = []
    log_names = []
    for ch in range(3):
        log_features.append(log2(window[ch]))
        log_names.append('log_{}'.format(ch))

    tr_features = []
    tr_names = []
    for ch in range(3):
        tr_features.append(time_reversibility(window[ch]))
        tr_names.append('tr_{}'.format(ch))

    ly_features = []
    ly_names = []
    for ch in range(3):
        ly_exp = lyapunov(window[ch])
        for i, exp in enumerate(ly_exp):
            ly_features.append(exp)
            ly_names.append('ly_ch{}_{}'.format(ch, i))

    si_features = []
    si_names = []
    for ch in range(3):
        si_features.append(simple_square_integral(window[ch]))
        si_names.append('si_{}'.format(ch))
            
    features = (tsfresh_features + corr_coefs + rms_features + sampen_features 
                + med_freq_features + peak_freq_features + log_features
                + tr_features + ly_features + si_features)
    names = (tsfresh_feature_names + corr_names + rms_names + sampen_names 
             + med_freq_feature_names + peak_freq_feature_names + log_names
             + tr_names + ly_names + si_names)

    queue.put((features, file, idx, names))


def extract_boss_features(train_windows, train_labels, test_windows):
    # TODO: Shouldn't we create out-of-sample features for the train_windows too?
    train_features = []
    test_features = []
    
    boss = BOSS(numerosity_reduction=False)
    for ch in range(3):
        X_train_ch = train_windows[:, ch, :]
        X_test_ch = test_windows[:, ch, :]
        boss.fit(X_train_ch, train_labels)

        train_features.append(boss.transform(X_train_ch).toarray())
        test_features.append(boss.transform(X_test_ch).toarray())

    train_features = np.hstack(train_features)
    test_features = np.hstack(test_features)

    return train_features, test_features


def process_header_file_tpehgt(file):
    start_idx = 0
    with open(file, 'r') as ifp:
        lines = ifp.readlines()
        for line_idx, line in enumerate(lines):
            if line.startswith('#'):
                start_idx = line_idx
                break
        
        names = []
        values = []
        for line in lines[start_idx+1:]:
            _, name, value = line.split()
            names.append(name)
            values.append(value)
            
        return names, values


def extract_clinical_features_tpehgt(files, DATA_DIR='tpehgts'):
    clinical_vars = []
    for file in tqdm(files, desc='Extracting clinical features...'):
        names, values = process_header_file_tpehgt('{}/{}.hea'.format(DATA_DIR, file))
        clinical_vars.append(values)

    clinical_df = pd.DataFrame(clinical_vars, columns=names)

    clinical_df = clinical_df.drop(['Gestation'], axis=1)
    clinical_df = clinical_df.replace('None', np.NaN)
    clinical_df = clinical_df.replace('N/A', np.NaN)
    clinical_df['ID'] = clinical_df['RecID']
    for col in ['Rectime', 'Age', 'Abortions', 'Weight']:
        clinical_df[col] = clinical_df[col].astype(float)
    clinical_df = clinical_df.drop_duplicates()

    return clinical_df[['ID', 'Rectime', 'Age', 'Parity', 'Abortions']]


def process_header_file_iceland(file):
    start_idx = 0
    with open(file, 'r') as ifp:
        lines = ifp.readlines()
        for line_idx, line in enumerate(lines):
            if line.startswith('#'):
                start_idx = line_idx
                break
        
        names = []
        values = []
        for line in lines[start_idx+1:]:
            if (line.count(':') == 1 and 'Comment' not in line 
                and 'contraction' not in line):
                name, value = line.split(':')
                names.append(name.strip('#'))
                values.append(value.strip())
            
        return names, values


def extract_clinical_features_iceland(files):
    pass


def extract_all_features(train_fit_windows, train_fit_labels, train_idx,
                         train_fit_files, test_windows, test_idx, test_files, 
                         DATA_DIR='tpehgts', 
                         clin_extract=extract_clinical_features_tpehgt):
    
    # Extract BOSS features
    print(end='Extracting BOSS features...  ')
    train_boss, test_boss = extract_boss_features(
        np.array(train_fit_windows), train_fit_labels, 
        np.array(test_windows)
    )
    print('OK!')

    _columns = ['boss_{}'.format(i) for i in range(len(train_boss[0]))]
    train_boss_df = pd.DataFrame(train_boss, columns=_columns)
    test_boss_df = pd.DataFrame(test_boss, columns=_columns)
    boss_df = pd.concat([train_boss_df, test_boss_df])

    # Extract unsupervised features from literature and using TSFRESH
    m = multiprocessing.Manager()
    res_queue = m.Queue()
    feature_procs = []
    all_windows = train_fit_windows + test_windows
    all_idx = train_idx + test_idx
    all_files = train_fit_files + test_files
    p = MyPool(n_cpu)

    p.map(partial(extract_window_features, queue=res_queue), 
          zip(all_windows, all_files, all_idx))
    p.close()
    p.join()

    feature_vectors = []
    while not res_queue.empty():
        features, file, idx, names = res_queue.get()
        vector = [file, idx] + features
        feature_vectors.append(vector)

    _columns = ['file', 'idx'] + names
    features_df = pd.DataFrame(feature_vectors, columns=_columns)
    features_df['file_idx'] = features_df['file'].apply(lambda x: all_files.index(x))
    features_df = features_df.sort_values(by=['file_idx', 'idx'])
    features_df = features_df.drop(['file_idx', 'idx'], axis=1)
    #print(features_df.sample(50))
    features_df = pd.concat([features_df.reset_index(drop=True),
                             boss_df.reset_index(drop=True)], axis=1)

    # Extract clinical variables
    clinical_df = clin_extract(set(train_fit_files + test_files))

    # Merge it all together
    ts_features = list(set(features_df.columns) - {'file'})
    clinical_features = list(set(clinical_df.columns) - {'ID'})
    features_df = features_df.merge(clinical_df, how='left', left_on='file', 
                                    right_on='ID')

    train_fit_features_df = features_df.iloc[:len(train_fit_windows)]
    #train_eval_features_df = features_df.iloc[len(train_fit_windows):(len(train_fit_windows) + len(train_eval_windows))]
    test_features_df = features_df.iloc[len(train_fit_windows):]

    return train_fit_features_df, test_features_df, ts_features, clinical_features
