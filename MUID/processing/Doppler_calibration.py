import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PCA import *
from dtw import *
import pickle
import os
from scipy.fft import fftshift
from dataPCA import PCA
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

    return spectrogram_all[:80, :]


def pca_spectrum(CSI_single, k):
    csi_pca = PCA(k).fit(CSI_single)
    matrix_sum = []
    for i in range(1, k):
        csi_component = csi_pca.components_[i]
        freq, times, spectrogram = signal.spectrogram(csi_component, 1000, window='hann', nperseg=wlen, noverlap=nlap,
                                                      nfft=1024, mode='magnitude', return_onesided=False)
        Zxx = np.abs(spectrogram)
        matrix_sum.append(Zxx)
    matrix_sum1 = np.sum(matrix_sum, axis=0)

    return matrix_sum1


def pca_spectrum_v2(CSI_single, k):
    csi_pca = pca(np.transpose(CSI_single), k)
    matrix_sum = []
    csi_pca = np.transpose(csi_pca)
    for i in range(1, k):
        csi_component = csi_pca[i]
        freq, times, spectrogram = signal.spectrogram(csi_component, 1000, window='hann', nperseg=wlen, noverlap=nlap,
                                                      nfft=1024, mode='magnitude', return_onesided=True)

        matrix_sum.append(spectrogram[:100, :])
    matrix_sum1 = np.sum(matrix_sum, axis=0)

    return matrix_sum1, freq[:100], times


def get_freq(spect, frequency):
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


def shape_calibration(v_acf, v_doppler, spectrum_pca):
    if v_acf.shape == v_doppler.shape:
        return v_acf, v_doppler, spectrum_pca
    else:
        v_doppler = v_doppler[:len(v_acf)]
        spectrum_pca = spectrum_pca[:, :len(v_acf)]

        return v_acf, v_doppler, spectrum_pca


def spectrum_calibration(v_acf, v_doppler_filter, stft_spectrum, frequency):
    v_acf = np.array([max(x, y) for x, y in zip(v_acf, v_doppler_filter)])
    alignment = dtw(v_acf, v_doppler_filter, window_type='slantedband',
                    window_args={"window_size": 30}, open_begin=False, open_end=False,
                    keep_internals=True)
    v_acf_align = np.array([v_acf[i] for i in alignment.index1])
    v_doppler_align = np.array([v_doppler_filter[i] for i in alignment.index2])

    ratio_dtw = np.array(v_doppler_align) / np.array(v_acf_align)
    ratio_dtw_filter = savgol_filter(ratio_dtw, 121, 3)
    x_intrep = np.linspace(0, len(ratio_dtw_filter), len(v_doppler_filter))
    ratio_dtw_filter_interp = np.interp(x_intrep, np.arange(len(ratio_dtw_filter)), ratio_dtw_filter)
    calibrated_spectrum = np.zeros_like(stft_spectrum)
    for i in range(stft_spectrum.shape[1]):
        scale_factor = ratio_dtw_filter_interp[i]
        scaled_frequencies = frequency * scale_factor
        calibrated_spectrum[:, i] = np.interp(scaled_frequencies, frequency, stft_spectrum[:, i])

    return calibrated_spectrum


if __name__ == '__main__':
    data_file = 'E:/Programs/PycharmProjects/multi_ac_monitor/envs/data_1_10_2p/'
    wlen = 400
    nlap = 396

    for multi_index in range(0, 1, 1):
        file_path = data_file + '4TrainTestData_10as_dir_reflect/'
        speed_file_path = data_file + '/5SpeedData_v3/'
        for perdir in os.listdir(file_path):
            if perdir.startswith("person"):
                train_data_path = os.path.join(file_path, perdir)
                speed_data_path = os.path.join(speed_file_path, perdir)

                temp_paths = os.listdir(train_data_path)
                temp_paths.sort(key=lambda x: int(x[11:-4]))

                for filename in temp_paths:
                    path = os.path.join(train_data_path, filename)
                    speed_path = os.path.join(speed_data_path, 'speed_data' + filename[11:])
                    with open(path, 'rb') as f:
                        CSI_image_index_flag = pickle.load(f)
                    CSI_image = CSI_image_index_flag[0]
                    CSI_index = CSI_image_index_flag[1]

                    CSI_dir = CSI_image[0, :, CSI_index[1]:CSI_index[2]]
                    CSI = CSI_image[1, :, CSI_index[1]:CSI_index[2]]
                    CSI_ratio = CSI / CSI_dir

                    CSI_amp = get_amp(CSI_ratio)
                    amp_rn = butterWorth_lowpass(CSI_amp, 600 / 150)
                    CSI_amp = butterWorth_bandpass(amp_rn, 500, 8, 40)

                    spectrum_pca, freq, time = pca_spectrum_v2(CSI_amp, 10)

                    f_torso, f_leg, f_spec_max = get_freq(spectrum_pca, freq)
                    v_torso = f_torso * 0.027

                    with open(speed_path, 'rb') as f:
                        speed_time = pickle.load(f)
                    v_acf = speed_time[0]
                    v_acf, v_torso, spectrum_pca = shape_calibration(v_acf, v_torso, spectrum_pca)
                    v_doppler_filter = savgol_filter(v_torso, 81, 2)

                    calibrated_spectrum = spectrum_calibration(v_acf, v_doppler_filter, spectrum_pca, freq)
                    spec_dtw = np.array([calibrated_spectrum])

                    # with open(train_data_cali_save_path + '/train_data' + filename[11:-4] + '.pkl', 'wb') as handle:
                    #     pickle.dump(spec_dtw, handle, protocol=4)
                    # print('person', perdir[-1], ' cali_data' + filename[11:-4] + 'stored')
