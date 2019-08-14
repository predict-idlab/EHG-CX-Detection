import os

import numpy as np

import wfdb

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


def read_signal(file_path, LOW_FREQ=0.05, HIGH_FREQ=0.3):
    record = wfdb.rdrecord(file_path)
    annotation = wfdb.rdann(file_path, 'atr')
    annotated_intervals = list(zip(annotation.sample, annotation.aux_note))
    
    signal_ch1 = record.p_signal[:, 0][:36000]
    signal_ch2 = record.p_signal[:, 2][:36000]
    signal_ch3 = record.p_signal[:, 4][:36000]
    
    signal_ch1 = butter_bandpass_filter(signal_ch1, LOW_FREQ, HIGH_FREQ, 20.0, order=4)
    signal_ch2 = butter_bandpass_filter(signal_ch2, LOW_FREQ, HIGH_FREQ, 20.0, order=4)
    signal_ch3 = butter_bandpass_filter(signal_ch3, LOW_FREQ, HIGH_FREQ, 20.0, order=4)

    signal_ch1 = medfilt(signal_ch1)
    signal_ch2 = medfilt(signal_ch2)
    signal_ch3 = medfilt(signal_ch3)

    #ch1_scaler = RobustScaler()
    #ch2_scaler = RobustScaler()
    #ch3_scaler = RobustScaler()

    #signal_ch1 = ch1_scaler.fit_transform(signal_ch1.reshape(-1, 1)).reshape(-1, )
    #signal_ch2 = ch2_scaler.fit_transform(signal_ch2.reshape(-1, 1)).reshape(-1, )
    #signal_ch3 = ch3_scaler.fit_transform(signal_ch3.reshape(-1, 1)).reshape(-1, )

    return signal_ch1, signal_ch2, signal_ch3, annotated_intervals


def read_signal_iceland(file_path, LOW_FREQ=0.05, HIGH_FREQ=0.3):
    record = wfdb.rdrecord(file_path)
    annotation = wfdb.rdann(file_path, 'atr')
    annotated_intervals = list(zip(annotation.sample, annotation.symbol))
    
    signal_ch1 = record.p_signal[:, 4] - record.p_signal[:, 0]
    signal_ch2 = record.p_signal[:, 4] - record.p_signal[:, 7]
    signal_ch3 = record.p_signal[:, 10] - record.p_signal[:, 7]
    
    signal_ch1 = butter_bandpass_filter(signal_ch1, LOW_FREQ, HIGH_FREQ, 200.0, order=4)
    signal_ch2 = butter_bandpass_filter(signal_ch2, LOW_FREQ, HIGH_FREQ, 200.0, order=4)
    signal_ch3 = butter_bandpass_filter(signal_ch3, LOW_FREQ, HIGH_FREQ, 200.0, order=4)

    signal_ch1 = medfilt(signal_ch1)
    signal_ch2 = medfilt(signal_ch2)
    signal_ch3 = medfilt(signal_ch3)
    
    signal_ch1 = resample(signal_ch1, 200, 20)
    signal_ch2 = resample(signal_ch2, 200, 20)
    signal_ch3 = resample(signal_ch3, 200, 20)

    #ch1_scaler = RobustScaler()
    #ch2_scaler = RobustScaler()
    #ch3_scaler = RobustScaler()

    #signal_ch1 = ch1_scaler.fit_transform(signal_ch1.reshape(-1, 1)).reshape(-1, )
    #signal_ch2 = ch2_scaler.fit_transform(signal_ch2.reshape(-1, 1)).reshape(-1, )
    #signal_ch3 = ch3_scaler.fit_transform(signal_ch3.reshape(-1, 1)).reshape(-1, )

    return signal_ch1, signal_ch2, signal_ch3, annotated_intervals


def extract_train_windows(file_path, window_size=1000, shift=1000, read_fn=read_signal, LOW_FREQ=0.05, HIGH_FREQ=0.3):
    signal_ch1, signal_ch2, signal_ch3, intervals = read_fn(file_path, LOW_FREQ=LOW_FREQ, HIGH_FREQ=HIGH_FREQ)
    indices = []
    windows = []
    labels = []
    
    for annotation1, annotation2 in zip(intervals[::2], intervals[1::2]):
        if annotation1[1][-1] not in ['C', 'D']:
            continue
          
        label = int(annotation1[1][-1] == 'C')

        if window_size is None:
            windows.append(
                  np.array([
                      signal_ch1[annotation1[0]:annotation2[0]],
                      signal_ch2[annotation1[0]:annotation2[0]],
                      signal_ch3[annotation1[0]:annotation2[0]],
                  ])
              )
            labels.append(label)
            indices.append(annotation1[0])
        else:
            for start in range(annotation1[0], annotation2[0] - window_size, shift):
                windows.append(
                    np.array([
                      signal_ch1[start:start+window_size],
                      signal_ch2[start:start+window_size],
                      signal_ch3[start:start+window_size],
                    ])
                )
                labels.append(label)
                indices.append(start)

    return windows, labels, indices


def extract_test_windows(file_path, window_size=1000, shift=1000, read_fn=read_signal, LOW_FREQ=0.05, HIGH_FREQ=0.3):
    signal_ch1, signal_ch2, signal_ch3, intervals = read_signal(file_path, LOW_FREQ=LOW_FREQ, HIGH_FREQ=HIGH_FREQ)
    indices = []
    windows = []

    for start in range(0, len(signal_ch1) - window_size, shift):
        windows.append(
            np.array([
              signal_ch1[start:start+window_size],
              signal_ch2[start:start+window_size],
              signal_ch3[start:start+window_size],
            ])
        )
        indices.append(start)

    return windows, indices
