import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import pylab
from processing.PCA import *
from dtw import *
import pickle
import os
from scipy.fft import fftshift
from processing.dataPCA import PCA
from scipy.signal import savgol_filter, find_peaks

fs = 1000
data = []
subcarrier_sum = 1


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

        return True
    else:
        return False


def butterWorth_lowpass(csimat, factor):
    b, a = signal.butter(5, 0.33334 / factor, 'lowpass')
    noise_remove = signal.filtfilt(b, a, csimat)
    return noise_remove


def butterWorth_bandpass(csimat, factor, f_low, f_high):
    b, a = signal.butter(5, [f_low / factor, f_high / factor], 'bandpass')
    noise_remove = signal.filtfilt(b, a, csimat)
    return noise_remove


def complexToLatitude(subcarrier):
    return np.round(abs(subcarrier), 4)


def get_amp(Matrix):
    amp = None
    for subcarrier in Matrix:
        amp = np.array([complexToLatitude(subcarrier)]) if amp is None else np.append(
            amp, [complexToLatitude(subcarrier)], axis=0)
    return amp


def get_phase(Matrix):
    phase = None
    for subcarrier in Matrix:
        phase = np.array([np.unwrap(np.angle(subcarrier))]) if phase is None else np.append(
            phase, [np.unwrap(np.angle(subcarrier))], axis=0)

    print(phase.shape)
    return phase


def norm_row(zp):
    data = np.array(zp)
    for i in range(0, len(data[0]), 1):
        data[:, i] = np.array(data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))
    return data


def subcarrier_spectrum_sum(CSI_single):
    spectrogram_all = []
    for i in range(30):
        freq, times, spectrogram = signal.spectrogram(CSI_single[i], 1000, window='hann', nperseg=wlen, noverlap=nlap,
                                                      nfft=1024, mode='magnitude', return_onesided=False)

        if len(spectrogram_all) == 0:
            spectrogram_all = spectrogram
        else:
            spectrogram_all += spectrogram
    spectrogram_norm = norm_row(spectrogram_all)

    return spectrogram_all[:80, :]


def pca_spectrum(CSI_single, k):
    csi_pca = pca(np.transpose(CSI_single), k)
    matrix_sum = []
    csi_pca = np.transpose(csi_pca)
    for i in range(1, k):
        csi_component = csi_pca[i]
        freq, times, spectrogram = signal.spectrogram(csi_component, 1000, window='hann', nperseg=wlen, noverlap=nlap,
                                                      nfft=1024, mode='magnitude', return_onesided=True)

        matrix_sum.append(spectrogram[:100, :])
    matrix_sum1 = np.sum(matrix_sum, axis=0)
    spectrogram_norm = norm_row(matrix_sum1)

    return spectrogram_norm, matrix_sum1, freq[:100], times


def get_freq(spect, frequency, times):
    f_upper = np.zeros([len(spect[0])])
    f_torso = []
    f_leg1 = []
    mag_sum = np.zeros([len(spect[0])])
    spect = np.array(spect)
    for i in range(0, len(spect[0]), 1):
        mag_sum[i] = np.sum(spect[:, i])
        if mag_sum[i] > 0:
            f_upper[i] = np.argmax(spect[:, i])

            for p in range(0, len(spect[:, 0]), 1):
                if np.sum(spect[0:p, i]) / mag_sum[i] > 0.35:
                    f_torso.append(frequency[p])
                    break
            for q in range(0, len(spect[:, 0]), 1):
                if np.sum(spect[0:q, i]) / mag_sum[i] > 0.8:
                    f_leg1.append(frequency[q])
                    break

        else:
            f_upper[i] = 0
            f_torso.append(0)
            f_leg1.append(0)

    f_upper = np.array(f_upper)
    f_torso = np.array(f_torso)
    f_leg1 = np.array(f_leg1)

    f_upper = savgol_filter(f_upper, 51, 2)
    f_torso = savgol_filter(f_torso, 41, 2)
    f_leg1 = savgol_filter(f_leg1, 41, 2)

    return f_torso, f_leg1, f_upper


def shape_calibration(v_acf, v_doppler, spectrum_pca, spectrum_pca_norm):
    if v_acf.shape == v_doppler.shape:
        return v_acf, v_doppler, spectrum_pca, spectrum_pca_norm

    else:
        v_doppler = v_doppler[:len(v_acf)]
        spectrum_pca = spectrum_pca[:, :len(v_acf)]
        spectrum_pca_norm = spectrum_pca_norm[:, :len(v_acf)]
        return v_acf, v_doppler, spectrum_pca, spectrum_pca_norm


def extract_features(v_tor, v_leg, spectrum_pca):
    features = []
    print(spectrum_pca.shape)
    peak, _ = find_peaks(v_tor, height=np.average(v_tor), distance=60)
    phase_feature_mat = np.zeros((len(peak) - 1, 4, 40))
    for i in range(1, len(peak) - 1):
        step = spectrum_pca[:40, peak[i]:peak[i + 1]]
        step_phases = [0, step.shape[1] // 4, step.shape[1] // 2, 3 * step.shape[1] // 4, step.shape[1] - 1]
        for j in range(len(step_phases) - 1):
            phases_spectra = step[:, step_phases[j]:step_phases[j + 1]]
            avg_phases_freq = np.mean(phases_spectra, axis=1)

            phase_feature_mat[i - 1, j, :] = avg_phases_freq
    phase_feature = np.mean(phase_feature_mat, axis=0)
    for row in phase_feature:
        features.extend(row)

    gait_cycles = []
    step_lengths = []
    for i in range(len(peak) - 1):
        peak_diff = (peak[i + 1] - peak[i])
        gait_cycles.append(peak_diff * 0.02)

        v_curve = v_tor[peak[i]:peak[i + 1]]
        step_length = np.trapz(v_curve, dx=0.02)
        step_lengths.append(step_length)

    gait_cycle_mean = np.mean(gait_cycles)

    step_length_mean = np.mean(step_lengths)

    v_tor_max, v_tor_min, v_tor_avg, v_tor_var, = np.max(v_tor), np.min(v_tor), np.average(v_tor), np.var(v_tor),
    v_leg_max, v_leg_min, v_leg_avg, v_leg_var, = np.max(v_leg), np.min(v_leg), np.average(v_leg), np.var(v_leg),
    features.extend(
        [gait_cycle_mean, step_length_mean, v_tor_max, v_tor_min, v_tor_avg, v_tor_var, v_leg_max, v_leg_min, v_leg_avg,
         v_leg_var])
    features = np.array(features)
    return features


if __name__ == '__main__':
    env = 'data_1_13_3p'
    data_file = 'E:/Programs/PycharmProjects/multi_ac_monitor/envs/' + env + '/'
    X_train, X_test, y_train, y_test = None, None, None, None
    count1 = 0
    wlen = 400
    nlap = 396

    for comb in os.listdir(data_file)[0:1]:
        for multi in range(0, 1):
            multi_path = data_file + 'multi_' + str(multi)
            file_path = data_file + '/4TrainTestData_10as_dir_reflect/'
            for perdir in os.listdir(file_path):
                if perdir.startswith("person"):
                    train_data_path = os.path.join(file_path, perdir)
                    train_data_save_path = 'E:/Programs/PycharmProjects/multi_ac_monitor/baseline/wifiu/' + env + \
                                           '/multi_' + str(multi) + '/person_' + str(perdir[-1])
                    mkdir(train_data_save_path)

                    temp_paths = os.listdir(train_data_path)
                    temp_paths.sort(key=lambda x: int(x[11:-4]))

                    for filename in temp_paths:
                        path = os.path.join(train_data_path, filename)
                        with open(path, 'rb') as f:
                            CSI_image_index_flag = pickle.load(f)
                        CSI_image = CSI_image_index_flag[0]  # (1*30*16101)
                        CSI_index = CSI_image_index_flag[1]

                        CSI_dir = CSI_image[0, :, CSI_index[1]:CSI_index[2]]  # (30*3826)
                        CSI = CSI_image[1, :, CSI_index[1]:CSI_index[2]]
                        CSI_ratio = CSI / CSI_dir

                        CSI_amp = get_amp(CSI_ratio)
                        amp_rn = butterWorth_lowpass(CSI_amp, 600 / 150)
                        CSI_amp = butterWorth_bandpass(amp_rn, 500, 8, 40)

                        spectrum_pca_norm, spectrum_pca, freq, time = pca_spectrum(CSI_amp, 10)

                        f_torso, f_leg, f_spec_max = get_freq(spectrum_pca, freq, time)
                        v_torso, v_leg, v_spec_max = f_torso * 0.027, f_leg * 0.027, f_spec_max * 0.027

                        features_wifiu = extract_features(v_torso, v_leg, spectrum_pca)

                        # with open(train_data_save_path + '/train_data' + filename[11:-4] + '.pkl', 'wb') as handle:
                        #     pickle.dump(np.array(features_wifiu), handle, protocol=4)
                        # print('person', perdir[-1], ' data' + filename[11:-4] + 'stored')
