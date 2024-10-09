import numpy as np
import pickle
import os
import math
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from processing.Speed_estimation import get_amp
from processing.Speed_estimation import mrc_acf
from processing.Doppler_calibration import butterWorth_lowpass

interval_time = 20


def normalize_test_features(features, train_min, train_max):
    normalized_features = (features - train_min) / (train_max - train_min)
    normalized_test_features = np.clip(normalized_features, 0, 1)
    return normalized_test_features


def normalize_train_features(features):
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    normalized_features = (features - min_values) / (max_values - min_values)
    return normalized_features, min_values, max_values


def calculate_percentile(velocity):
    mean_velocity = np.mean(velocity)
    v_sg = savgol_filter(velocity, 60, 3)
    velocity_deviation = velocity - v_sg
    positive_bias = velocity_deviation[velocity_deviation > 0]
    negative_bias = velocity_deviation[velocity_deviation < 0]
    absolute_bias = np.abs(velocity_deviation)

    positive_percentiles = np.percentile(positive_bias, [50, 75, 95])
    negative_percentiles = np.percentile(negative_bias, [50, 25, 5])
    absolute_percentiles = np.percentile(absolute_bias, [50, 75, 95])
    percentile_values = np.concatenate((positive_percentiles, negative_percentiles, absolute_percentiles))

    return percentile_values


def calculate_smoothness(peaks, acc):
    harmonic_ratios = []
    for i in range(len(peaks) - 1):
        dft_double_side = np.fft.fft(acc[peaks[i]:peaks[i + 1]])
        dft = dft_double_side[0:math.ceil(len(dft_double_side) / 2)]

        even_harmonics_sum = np.sum(np.abs(dft[2: 21:2]))
        odd_harmonics_sum = np.sum(np.abs(dft[1: 21:2]))
        if odd_harmonics_sum == 0:
            continue
        harmonic_ratio = even_harmonics_sum / odd_harmonics_sum
        if harmonic_ratio >= 30:
            continue
        harmonic_ratios.append(harmonic_ratio)
    hr_mean = np.median(harmonic_ratios)
    hr_var = np.var(harmonic_ratios)
    return hr_mean, hr_var


def calculate_rhythmicity(v):
    window_len, window_step, nlag = 60, 20, 60
    rhythmicity = []
    first_peak_heights = []
    num_significant_peaks = []
    count1, count2 = 0, 0
    for t in range(0, len(v) - window_len, window_step):
        acf_speed = acf(v[t:t + window_len], nlags=nlag)
        peaks, _ = find_peaks(acf_speed, prominence=0.05, height=0)

        if len(peaks) > 0:
            first_peak_heights.append(acf_speed[peaks[0]])
            num_significant_peaks.append(len(peaks))
            count1 += 1
        else:
            count2 += 1
    if first_peak_heights:
        avg_first_peak_height = np.mean(first_peak_heights)
        var_first_peak_height = np.var(first_peak_heights)
    else:
        avg_first_peak_height = np.zeros(1)
        var_first_peak_height = np.zeros(1)
    ratio = np.array(count2 / (count1 + count2))

    rhythmicity.extend([avg_first_peak_height, var_first_peak_height, ratio])
    return rhythmicity


def calculate_acf_feature(peaks, comb_index, multi_index, person_index, data_index):
    peak_acf_diff_list = []
    window_len, window_step, nlag = 450, 20, 50

    filename = 'E:/Programs/PycharmProjects/multi_ac_monitor/data_10_24_2p/sample_rate_1000/comb' + str(comb_index) + \
               '/multi_' + str(multi_index) + '/4TrainTestData_10as/person_' + str(person_index) + '/train_data_' + str(
        data_index) + '.pkl'
    with open(filename, 'rb') as f:
        CSI_image_index_flag = pickle.load(f)
    CSI_image = CSI_image_index_flag[0]
    CSI_index = CSI_image_index_flag[1]
    CSI = CSI_image[0, :, CSI_index[1]:CSI_index[2]]

    CSI_amp = get_amp(CSI)
    amp_rn = butterWorth_lowpass(CSI_amp, 600 / 150)
    peaks_times = peaks * 20
    for peaks_time in peaks_times:
        peak_acf = mrc_acf(amp_rn[:, peaks_time:peaks_time + window_len], nlag)
        peak_acf_diff = np.diff(peak_acf)
        peak_acf_diff_list.append(peak_acf_diff[0:50])

    average_acf = np.average(peak_acf_diff_list, axis=0)
    acf_feature = average_acf

    return acf_feature


def extract_gait_features(velocity, comb_index, multi_index, person_index, data_index):
    peaks, _ = find_peaks(velocity, distance=10, height=0.5)
    gait_cycles = []
    step_lengths = []
    for i in range(len(peaks) - 1):
        peak_diff = (peaks[i + 1] - peaks[i])
        gait_cycles.append(peak_diff * 0.02)

        v_curve = velocity[peaks[i]:peaks[i + 1]]
        step_length = np.trapz(v_curve, dx=0.02)
        step_lengths.append(step_length)

    gait_cycle_mean = np.mean(gait_cycles)
    gait_cycle_var = np.var(gait_cycles)
    step_length_mean = np.mean(step_lengths)
    step_length_var = np.var(step_lengths)

    acc = np.diff(velocity)
    avg_acc = np.mean(acc)
    acc_max = np.max(acc)
    acc_min = np.min(acc)
    acc_var = np.var(acc)

    percentile_values = calculate_percentile(velocity)

    hr_mean, hr_var = calculate_smoothness(peaks, acc)

    rhythmicity = calculate_rhythmicity(velocity)

    acf_feature = calculate_acf_feature(peaks, comb_index, multi_index, person_index, data_index)

    features = [gait_cycle_mean, gait_cycle_var, step_length_mean, step_length_var, avg_acc, acc_max, acc_min, acc_var]
    features.extend(percentile_values)
    features.extend([hr_mean, hr_var])
    features.extend(rhythmicity)
    features.extend(acf_feature)
    return np.array(features)


def extract_gait_data(folder):
    features_list = []
    labels = []
    count = 0
    for comb in os.listdir(folder):
        comb_path = os.path.join(folder, comb)
        for multi in os.listdir(comb_path):
            multi_path = os.path.join(comb_path, multi)
            speed_data_path = multi_path + '/5SpeedData'
            for subdir in os.listdir(speed_data_path)[0:2]:
                if subdir.startswith("person"):
                    label = int(subdir.split("_")[1])
                    subdir_path = os.path.join(speed_data_path, subdir)
                    temp_paths = os.listdir(subdir_path)
                    temp_paths.sort(key=lambda x: int(x[10:-4]))
                    for file in temp_paths:
                        if file.endswith(".pkl"):
                            data_path = os.path.join(subdir_path, file)
                            with open(data_path, 'rb') as f:
                                velocity_data = pickle.load(f)
                            if len(velocity_data.shape) == 2:
                                features_list.append(
                                    extract_gait_features(velocity_data[0], comb[-1], multi[-1], label, file[10:-4]))
                                labels.append(label)
                                count += 1
                            else:
                                print('velocity shape is wrong! It is:', velocity_data.shape)

    return np.array(features_list), np.array(labels)


if __name__ == '__main__':
    speed_file = 'E:/Programs/PycharmProjects/multi_ac_monitor/data_10_24_2p/sample_rate_1000/'
    X, y = extract_gait_data(speed_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    normalized_X_train, min_vals, max_vals = normalize_train_features(X_train)
    normalized_X_test = normalize_test_features(X_test, min_vals, max_vals)

    clf = svm.SVC(kernel='rbf', C=1)
    clf.fit(normalized_X_train, y_train)

    y_train_pred = clf.predict(normalized_X_train)
    y_test_pred = clf.predict(normalized_X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    print("-----Accuracy of test set is:", accuracy)

    cm = confusion_matrix(y_test, y_test_pred)
    print("-----Confusion Matrix:")
    print(cm)
