import numpy as np
import pickle
from processing import PCA
from matplotlib import pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import acf
import os
from scipy.spatial.distance import cosine, euclidean
from scipy.special import kl_div
from scipy.signal import savgol_filter, find_peaks


lumbuda = 2 * 2.727e-2
d = 0.03  # distance between adjacent antennas in the linear antenna array

tx = 1
rx = 7  # number of rx antennas
sub_carr_num = 30
# fs = 20e6  # channel bandwidth
c = 3e8  # speed of light
cal_angel = 48.4


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def complexToLatitude(subcarrier):
    return np.round(abs(subcarrier), 4)


def complexToPower(subcarrier):
    return np.round(pow(abs(subcarrier), 2), 4)


def get_power(pair):
    power = None
    for subcarrier in pair:
        power = np.array([complexToPower(subcarrier)]) if power is None else np.append(
            power, [complexToPower(subcarrier)], axis=0)
    return power


def get_CSIMatrix(CSI_data):
    CSIMatrix = None
    for windows in CSI_data:
        # print(csi.shape)
        CSIMatrix = np.array(windows) if CSIMatrix is None else np.append(
            CSIMatrix, windows, axis=0)
    print(CSIMatrix.shape)
    CSIMatrix = np.transpose(CSIMatrix, [1, 2, 0])
    return CSIMatrix


def pca_CSI_Matrix(CSI_Matrix, num_components):
    CSI_Matrix = np.transpose(CSI_Matrix, [0, 2, 1])
    CSI_pca = None
    for pair in CSI_Matrix:  # sub PCA and denoise
        PCA_csi = PCA.pca(pair, num_components)  # (5000,5)
        PCA_csi = np.sum(PCA.pca(pair, num_components), axis=1)

        CSI_pca = np.array([PCA_csi]) if CSI_pca is None else np.append(
            CSI_pca, [PCA_csi], axis=0)

    return CSI_pca


def avg_CSI_Matrix(CSI_Matrix):
    CSI_avg = None
    for pair in CSI_Matrix:  # sub PCA and denoise
        avg_csi = np.mean(pair, axis=0)

        CSI_avg = np.array([avg_csi]) if CSI_avg is None else np.append(
            CSI_avg, [avg_csi], axis=0)
    return CSI_avg


def calculate_cmov(csi_mat):
    CSI_power = None
    for pair in csi_mat:
        power = get_power(pair)
        CSI_power = np.array([power]) if CSI_power is None else np.append(
            CSI_power, [power], axis=0)

    power_spectrum = CSI_power
    average_spectrum = np.mean(power_spectrum, axis=2)
    cavg_matrix = np.tile(average_spectrum[:, :, np.newaxis], (1, power_spectrum.shape[2]))
    cmov_matrix = power_spectrum - cavg_matrix
    return cmov_matrix


def project_to_spectrum(AoA_mat, CSI_mov, window_size=500, antenna_num=6):
    """hermite"""

    def hermite_poly(k, t):
        if k == 0:
            return np.exp(-t ** 2 / 2) / pow(np.pi, 1 / 4)
        elif k == 1:
            return np.sqrt(2) * t / (pow(np.pi, 1 / 4) * np.exp(-t ** 2 / 2))
        else:
            return t * np.sqrt(2 / k) * hermite_poly(k - 1, t) - np.sqrt(k - 1 / k) * hermite_poly(k - 2, t)

    def hermite_window(input_signal, start_time, end_time, order):
        t = np.linspace(start_time / 1000, end_time / 1000, len(input_signal))
        window = hermite_poly(order, t)
        windowed_signal = input_signal * window
        return windowed_signal

    spectrograms = []

    for AoAs_individual in AoA_mat[1:, :]:
        projections = []
        t_end = 0
        AoA_consider = np.cos(42 / 180 * np.pi) - np.cos(AoAs_individual / 180 * np.pi)
        for i in range(len(CSI_mov)):
            phase = np.exp([-1j * 2 * np.pi / lumbuda * AoA_consider[i] * l * d for l in range(0, antenna_num)])
            length = len(CSI_mov[i])
            phase = np.tile(phase, (length, 1))
            projection = np.trapz(np.multiply(CSI_mov[i], phase), axis=1, dx=lumbuda / 2)
            projections.extend(projection)

        projections = np.array(projections).reshape(-1)
        freq, times, spectrogram_individual = signal.spectrogram(projections, 1000, nperseg=600, noverlap=596,
                                                                 window='cosine')

        freq_small = np.concatenate((freq[-30:], freq[:30]))
        spectrogram_individual_small = np.concatenate((spectrogram_individual[-30:, :], spectrogram_individual[:30, :]))

        spectrograms.append(spectrogram_individual_small)

    return spectrograms, freq_small


def extract_features(spectrogram, frequency):
    def get_freq(spect, plot=0):
        f_upper = np.zeros([len(spect[0])])
        f_torso = []
        f_leg1 = []
        mag_sum = np.zeros([len(spect[0])])
        spect = np.array(spect)
        for i in range(0, len(spect[0]), 1):
            mag_sum[i] = np.sum(spect[:, i])

            if mag_sum[i] > 0:
                for p in range(0, len(spect[:, 0]), 1):
                    if np.sum(spect[0:p, i]) / mag_sum[i] > 0.5:
                        f_torso.append(frequency[p])
                        break
                for q in range(0, len(spect[:, 0]), 1):
                    if np.sum(spect[0:q, i]) / mag_sum[i] > 0.9:
                        f_leg1.append(frequency[q])
                        break

                for j in range(len(spect[:, 0]) - 1, -1, -1):
                    if spect[j, i] / mag_sum[i] > 0.05:
                        f_upper[i] = (frequency[j])
                        break

            else:
                f_torso.append(0)
                f_leg1.append(0)
                f_upper[i] = 0
        f_upper = np.array(f_upper)
        f_torso = np.array(f_torso)
        f_leg1 = np.array(f_leg1)

        f_leg1 = savgol_filter(f_leg1, 100, 2)
        f_upper = savgol_filter(f_upper, 100, 2)
        f_torso = savgol_filter(f_torso, 100, 2)

        return f_torso, f_leg1, f_upper

    features = []
    # frequency feature
    spectrogram_mov = spectrogram[int(spectrogram.shape[0] / 2):, :]
    num_freq_bins = spectrogram_mov.shape[0]

    avg_freq = np.mean(spectrogram_mov[:, :], axis=1)
    features.append(avg_freq)
    avg_freq_norm = avg_freq / np.sum(avg_freq)

    f_torso, f_leg, f_upper = get_freq(spectrogram_mov, plot=0)
    peak, _ = find_peaks(f_upper, height=10, distance=60, prominence=0.2)

    phase_feature_mat = np.zeros((len(peak) - 1, 4, spectrogram_mov.shape[0]))
    for i in range(1, len(peak) - 1):
        step = spectrogram_mov[:, peak[i]:peak[i + 1]]
        step_phases = [0, step.shape[1] // 4, step.shape[1] // 2, 3 * step.shape[1] // 4, step.shape[1] - 1]
        for j in range(len(step_phases) - 1):
            phases_spectra = step[:, step_phases[j]:step_phases[j + 1]]
            avg_phases_freq = np.mean(phases_spectra, axis=1)

            phase_feature_mat[i - 1, j, :] = avg_phases_freq
    phase_feature = np.mean(phase_feature_mat, axis=0)
    for row in phase_feature:
        features.append(row)

    v_leg = f_leg * lumbuda / 2
    v_torso = f_torso * lumbuda / 2
    avg_leg_speed = np.mean(v_leg)
    avg_torso_speed = np.mean(v_torso)

    features.append(avg_leg_speed)
    features.append(avg_torso_speed)

    # temporal feature
    max_lag = 200

    autocorr = np.zeros((num_freq_bins, max_lag + 1))
    for i in range(num_freq_bins):
        autocorr[i] = acf(spectrogram_mov[i], nlags=max_lag)
    average_autocorr = np.sum(autocorr * avg_freq_norm[:, np.newaxis], axis=0)
    features.append(average_autocorr)

    percentile_50 = np.percentile(spectrogram_mov, 50, axis=0)
    percentile_70 = np.percentile(spectrogram_mov, 70, axis=0)
    corr_percentile_50 = acf(percentile_50, nlags=max_lag)
    corr_percentile_70 = acf(percentile_70, nlags=max_lag)

    features.append(corr_percentile_50)
    features.append(corr_percentile_70)

    return features


def calculate_feature_distances(query_features, candidate_features):
    num_features = len(query_features)
    feature_distances = []
    for i in range(num_features):
        query_feature = query_features[i]
        candidate_feature = candidate_features[i]
        if i < 5:
            distance = kl_div(query_feature, candidate_feature).sum()
        elif i == 5 or i == 6:
            distance = euclidean(query_feature, candidate_feature)
        else:
            distance = 1 - cosine(query_feature, candidate_feature)
        feature_distances.append(distance)

    return feature_distances


if __name__ == '__main__':
    for comb_index in range(1, 4, 1):
        for index in range(0, 1, 1):
            savePath = 'E:/Programs/PycharmProjects/multi_ac_monitor/envs/data_1_10_3p/3CSI_Tracking_AoA/'
            CSI_Tracking_AoA_SavePath = savePath + 'comb_' + str(comb_index) + '/'
            temp_paths = os.listdir(CSI_Tracking_AoA_SavePath)
            temp_paths.sort(key=lambda x: int(x[15:-4]))
            for filename in temp_paths:
                with open(CSI_Tracking_AoA_SavePath + filename, 'rb') as f:
                    CSIandAoA = pickle.load(f)

                # TrainingData_SavePath = 'E:/Programs/PycharmProjects/multi_ac_monitor/baseline/data_1_10_3p/features_v3_low'
                # mkdir(TrainingData_SavePath)

                CSI_data = CSIandAoA[0]
                AoA_matrix = CSIandAoA[1]

                CSIMatrix = get_CSIMatrix(CSI_data)
                CSI_len = CSIMatrix.shape[-1]
                AoA_matrix = AoA_matrix + cal_angel

                CSIMatrix = CSI_data[7:32]

                CSI_mov_mat = []
                for CSI_mat in CSIMatrix:
                    CSI_mat = np.transpose(CSI_mat, (1, 2, 0))
                    CSI_mov = calculate_cmov(CSI_mat)

                    CSI_mov = pca_CSI_Matrix(CSI_mov, 1)
                    CSI_mov = np.transpose(CSI_mov)
                    CSI_mov_mat.append(CSI_mov)

                aoa_tracking_mat = AoA_matrix[:, 6:32]
                spectrograms_all, freq = project_to_spectrum(aoa_tracking_mat, CSI_mov_mat, 200,
                                                             7)  # AoA, CSI_mov, window_size,antenna_num

                activity_index_flag = []
                for i in range(2, len(CSIandAoA) - 2):
                    activity_index_flag.append(CSIandAoA[i])
                person = int(len(activity_index_flag) / 3)
                CSI_index_list = []
                for j in range(person):
                    features_individual = extract_features(spectrograms_all[j], freq[int(len(freq) / 2):])

                    # TrainData_savePath_person = TrainingData_SavePath + '/person_' + str(
                    #     activity_index_flag[j * 3 + 2]) + '/'
                    # mkdir(TrainData_savePath_person)
                    #
                    # with open(TrainData_savePath_person + 'train_data' + filename[14:],
                    #           'wb') as handle:
                    #     pickle.dump(features_individual, handle, protocol=4)
