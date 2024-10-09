import numpy as np
import pickle
import os
from statsmodels.tsa.stattools import acf
from Speed_estimation import get_amp
from Doppler_calibration import butterWorth_lowpass


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def mrc_acf(power_matrix, lag):
    acf_matrix = []
    weight = []
    for subcarrier in power_matrix:
        acf_subcarrier = acf(subcarrier, nlags=lag)
        weight.append(acf_subcarrier[1])
        acf_matrix.append(acf_subcarrier)
    acf_mrc = np.average(acf_matrix, axis=0, weights=weight)
    return acf_mrc


def normalization(acf):
    max_acf = np.max(acf)
    min_acf = np.min(acf)
    acf = (acf - min_acf) / (max_acf - min_acf)
    return acf


def time_reshape(train_data):
    time_length = train_data.shape[1]
    freq_length = train_data.shape[0]
    interpolated_data = np.zeros((freq_length, 800))
    for f in range(freq_length):
        interpolated_data[f, :] = np.interp(np.linspace(0, time_length, 800), range(time_length), train_data[f, :])
    return interpolated_data


if __name__ == '__main__':
    data_file = 'E:/Programs/PycharmProjects/multi_ac_monitor/envs/data_1_10_2p/'
    sample_rate = 1000
    for multi_index in range(0, 1, 1):
        train_data_path = data_file + '/4TrainTestData_10as_dir_reflect'
        for perdir in os.listdir(train_data_path):
            if perdir.startswith("person"):
                sub_dir = os.path.join(train_data_path, perdir)
                temp_paths = os.listdir(sub_dir)
                temp_paths.sort(key=lambda x: int(x[11:-4]))
                acf_save_path = data_file + '/7acf_feature_mrc_100/' + '/person_' + perdir[-1]
                mkdir(acf_save_path)

                for file in temp_paths:
                    if file.endswith(".pkl"):
                        filename = os.path.join(sub_dir, file)
                        with open(filename, 'rb') as f:
                            CSI_image_index_flag = pickle.load(f)
                        CSI_image = CSI_image_index_flag[0]
                        CSI_index = CSI_image_index_flag[1]

                        CSI_dir = CSI_image[0, :, CSI_index[1]:CSI_index[2]]
                        CSI = CSI_image[1, :, CSI_index[1]:CSI_index[2]]
                        CSI_ratio = CSI / CSI_dir

                        CSI_amp = get_amp(CSI_ratio)
                        amp_rn = butterWorth_lowpass(CSI_amp, 600 / 150)
                        window_len, nlag = 100, 100

                        t = 0
                        win_acf_diff_list = []
                        while (t + window_len) < len(amp_rn[0]):
                            acf_mrc = mrc_acf(amp_rn[:, t:t + window_len + 1], nlag)
                            acf_diff_w_m = np.diff(acf_mrc)
                            win_acf_diff_list.append(acf_diff_w_m)
                            t += 4
                        x = np.arange(len(win_acf_diff_list)) / 1000
                        y = np.arange(len(win_acf_diff_list[0])) / 1000
                        acf_feature = np.array(win_acf_diff_list).transpose()

                        # with open(acf_save_path + '/train_data' + file[11:-4] + '.pkl',
                        #           'wb') as handle:
                        #     pickle.dump(np.array(acf_feature), handle, protocol=4)
                        # print('person', perdir[-1], ' one data stored')
