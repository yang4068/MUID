import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import scipy.signal
from statsmodels.tsa.stattools import acf
# from Preprocess import get_csiAmp
from processing.dataPCA import PCA
from scipy.signal import savgol_filter
from processing.Doppler_calibration import butterWorth_lowpass
import os

sample_rate = 1000
windowlen = 450
nlags = 200
interval_time = 4
pca_num = 1
pca_start_num = 0


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


def find_peak(acf, last_index):
    peakindex = 0
    if not last_index:
        for index in range(20, len(acf)):
            if index <= len(acf) - 6:
                if acf[index + 5] < acf[index + 4] < acf[index + 3] < acf[index + 2] < acf[index + 1] < acf[index] > \
                        acf[index - 1] > acf[index - 2] > acf[index - 3] > acf[index - 4] > acf[index - 5]:
                    peakindex = index
                    break
            elif index <= len(acf) - 2:
                if acf[index - 1] < acf[index] > acf[index + 1]:
                    peakindex = index
                    break
            else:
                peakindex = len(acf) - 1
    else:
        for index in range(20, len(acf)):
            if index <= len(acf) - 6:
                if acf[index + 5] < acf[index + 4] < acf[index + 3] < acf[index + 2] < acf[index + 1] < acf[index] > \
                        acf[index - 1] > acf[index - 2] > acf[index - 3] > acf[index - 4] > acf[index - 5]:
                    peakindex = index
                    if last_index[-1] > 75 and acf[index] < 0:
                        if last_index[-1] - index > 50:
                            continue
                    break
            elif index <= len(acf) - 2:
                if acf[index - 1] < acf[index] > acf[index + 1]:
                    peakindex = index
                    break
            else:
                peakindex = len(acf) - 1
    return peakindex


def average_acf(power_matrix, lag):
    acf_matrix = []
    for subcarrier in power_matrix:
        acf_matrix.append(acf(subcarrier, nlags=lag))
    acf_average = np.average(acf_matrix, axis=0)

    return acf_average


def mrc_acf(power_matrix, lag):
    acf_matrix = []
    weight = []
    for subcarrier in power_matrix:
        acf_subcarrier = acf(subcarrier, nlags=lag)
        weight.append(acf_subcarrier[1])
        acf_matrix.append(acf_subcarrier)
    acf_mrc = np.average(acf_matrix, axis=0, weights=weight)
    return acf_mrc


def pca_acf(power_matrix, lag):
    csi_pca_data = PCA(pca_start_num + pca_num).fit(power_matrix[0:30, :])
    pca_CSI_sum = []
    for i in range(pca_start_num, pca_start_num + pca_num):
        pca_CSI = csi_pca_data.components_[i]
        pca_CSI_sum.append(pca_CSI)
    pca_CSI_pw = np.sum(pca_CSI_sum, axis=0) / pca_num
    acf_pca = acf(pca_CSI_pw, nlags=lag)

    return acf_pca


if __name__ == '__main__':
    for multi_index in range(0, 1, 1):
        path = 'E:/Programs/PycharmProjects/multi_ac_monitor/envs/data_1_10_2p/'
        file_path = path + '/4TrainTestData_10as_dir_reflect/'
        for subdir in os.listdir(file_path):
            if subdir.startswith("person"):
                person_index = int(subdir.split("_")[1])
                subdir_path = os.path.join(file_path, subdir)
                temp_paths = os.listdir(subdir_path)
                temp_paths.sort(key=lambda x: int(x[11:-4]))

                Speedpic_savePath_person = path + '/5SpeedData_pic_v3/' + '/person_' + str(person_index)
                SpeedData_savePath_person = path + '/5SpeedData_v3/' + '/person_' + str(person_index)
                mkdir(Speedpic_savePath_person)
                mkdir(SpeedData_savePath_person)
                for train_data in temp_paths:
                    train_data_path = os.path.join(subdir_path, train_data)
                    with open(train_data_path, 'rb') as f:
                        CSI_image_index_flag = pickle.load(f)
                    CSI_image = CSI_image_index_flag[0]
                    CSI_index = CSI_image_index_flag[1]
                    CSI_dir = CSI_image[0, :, CSI_index[1]:CSI_index[2]]
                    CSI = CSI_image[1, :, CSI_index[1]:CSI_index[2]]
                    CSI_ratio = CSI / CSI_dir

                    length = CSI.shape[1]

                    CSI_ratio_amp = get_amp(CSI_ratio)
                    amp_rn_ratio = butterWorth_lowpass(CSI_ratio_amp, 600 / 150)
                    csi_ratio_pw = pow(amp_rn_ratio, 2)
                    CSI_ratio_pw = np.reshape(csi_ratio_pw, (-1, length))

                    ti = []
                    peaks = []

                    for t in range(0, length - windowlen, interval_time):
                        acf_org = mrc_acf(CSI_ratio_pw[:, t:t + windowlen], nlags)
                        acf_diff = np.diff(acf_org)
                        acf_diff_filter = scipy.signal.savgol_filter(acf_diff, 61, 2)
                        peak_index = find_peak(acf_diff_filter, peaks)
                        peaks.append(peak_index)
                        ti.append((t + windowlen / 2) / 1000)

                    vi = [0.54 * 0.0545 / (i / sample_rate) for i in peaks]
                    vi_filter = savgol_filter(vi, 61, 3)
                    speed_time = [np.array(vi_filter), np.array(vi), np.array(ti)]

                    # with open(SpeedData_savePath_person + '/speed_data' + train_data[11:-4] + '.pkl', 'wb') as handle:
                    #     pickle.dump(np.array(speed_time), handle, -1)
                    # print('person', str(person_index), ' one data stored')
