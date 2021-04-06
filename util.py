import wfdb
from scipy.signal import butter

<<<<<<< HEAD
from scipy.signal import butter, lfilter, medfilt
from librosa.core import resample

from sklearn.preprocessing import RobustScaler

def partition_data(directory, n_splits=5):
    files = set([x.split('.')[0] for x in os.listdir(directory)])
    p_files, t_files, n_files = [], [], []
    for file in files:
        if file[-4] == 'n':
            n_files.append(file)
        elif file[-4] == 'p':
            p_files.append(file)
        else:
            t_files.append(file)

    np.random.shuffle(p_files)
    np.random.shuffle(t_files)

    folds = []
    for split in range(n_splits):
        start = lambda x: int(x * (split / n_splits))
        end   = lambda x: int(x * ((split + 1) / n_splits))
        if split == n_splits - 1:
            test_p_files = p_files[start(len(p_files)):]
            test_t_files = t_files[start(len(t_files)):]
        else:
            test_p_files = p_files[start(len(p_files)):end(len(p_files))]
            test_t_files = t_files[start(len(t_files)):end(len(t_files))]

        train_p_files = list(set(p_files) - set(test_p_files))
        train_t_files = list(set(t_files) - set(test_t_files))

        test_files = test_t_files + test_p_files
        train_files = train_t_files + train_p_files

        folds.append((train_files, test_files))

    return folds

=======
import numpy as np
>>>>>>> 14f34a42457eff7b688b6ddbf28e846506543c84

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def _read_signal(file):
    record = wfdb.rdrecord(file_path)
    annotation = wfdb.rdann(file_path, 'atr')
    annotated_intervals = list(zip(annotation.sample, annotation.aux_note))
    
    signal_ch1 = record.p_signal[:, 0][3000:-3000]
    signal_ch2 = record.p_signal[:, 2][3000:-3000]
    signal_ch3 = record.p_signal[:, 4][3000:-3000]
    
    signal_ch1 = butter_bandpass_filter(signal_ch1, self.low_freq, 
                                        self.high_freq, sample_freq, order=4)
    signal_ch2 = butter_bandpass_filter(signal_ch2, self.low_freq, 
                                        self.high_freq, sample_freq, order=4)
    signal_ch3 = butter_bandpass_filter(signal_ch3, self.low_freq, 
                                        self.high_freq, sample_freq, order=4)

    return signal_ch1, signal_ch2, signal_ch3, annotated_intervals


def extract_windows(signals, start, end, window_size, window_shift):
	windows = []
	indices = []
	for start in range(start, min(end, len(signals[0])) - window_size, window_shift):
		windows.append([signal[start:start + window_size] for signal in signals])
		indices.append(start)
	return windows, indices


def extract_all_windows(file, window_size, window_shift, read_fn, low_freq, high_freq, sample_freq, labels=True):
    signal_ch1, signal_ch2, signal_ch3, intervals = read_fn(file, low_freq, high_freq, sample_freq)
    ch1_min, ch1_max = np.percentile(signal_ch1, 10), np.percentile(signal_ch1, 90)
    ch2_min, ch2_max = np.percentile(signal_ch2, 10), np.percentile(signal_ch2, 90)
    ch3_min, ch3_max = np.percentile(signal_ch3, 10), np.percentile(signal_ch3, 90)
    
    norm_signal_ch1 = (signal_ch1 - ch1_min) / (ch1_max - ch1_min)
    norm_signal_ch2 = (signal_ch2 - ch2_min) / (ch2_max - ch2_min)
    norm_signal_ch3 = (signal_ch3 - ch3_min) / (ch3_max - ch3_min)

    signals = [signal_ch1, signal_ch2, signal_ch3]
    norm_signals = [norm_signal_ch1, norm_signal_ch2, norm_signal_ch3]

    if labels:
        norm_windows, windows, labels, indices = [], [], [], []
        for ann1, ann2 in zip(intervals[::2], intervals[1::2]):
            if ann1[1][-1] not in ['C', 'D'] or ann2[0] >= len(signal_ch1) or ann1[0] < 0:
                continue
              
            label = int(ann1[1][-1] == 'C')
            interval_windows, idx = extract_windows(signals, ann1[0], ann2[0], window_size, window_shift)
            interval_norm_windows, _ = extract_windows(norm_signals, ann1[0], ann2[0], window_size, window_shift)
            windows.extend(interval_windows)
            norm_windows.extend(interval_norm_windows)
            labels.extend([label]*len(interval_windows))
            indices.extend(idx)

        return np.array(norm_windows), np.array(windows), np.array(labels), np.array(indices)
    else:
        windows, indices = extract_windows(signals, 0, len(signals[0]), window_size, window_shift)
        norm_windows, _ = extract_windows(norm_signals, 0, len(signals[0]), window_size, window_shift)
        return np.array(norm_windows), np.array(windows), np.array(indices)