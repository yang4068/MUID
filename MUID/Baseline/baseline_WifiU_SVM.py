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


if __name__ == '__main__':
    data_file = 'E:/Programs/PycharmProjects/multi_ac_monitor/baseline/wifiu/data_1_13_3p/'
    X_train, X_test, y_train, y_test = None, None, None, None
    count = 0
    for multi in os.listdir(data_file)[0:1]:
        multi_path = os.path.join(data_file, multi)

        for perdir in os.listdir(multi_path):
            if perdir.startswith("person"):
                label = int(perdir.split("_")[1])
                subdir_path = os.path.join(multi_path, perdir)
                temp_paths = os.listdir(subdir_path)
                temp_paths.sort(key=lambda x: int(x[10:-4]))
                features_list = []
                y=[]
                for file in temp_paths:
                    if file.endswith(".pkl"):
                        data_path = os.path.join(subdir_path, file)
                        with open(data_path, 'rb') as f:
                            feature_data = pickle.load(f)
                        features_list.append(feature_data)
                        y.append(label)
                X = features_list

                np.random.seed(98)
                np.random.shuffle(X)

                X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, shuffle=False, test_size=0.3)
                X_train = X1_train if X_train is None else np.append(X_train, X1_train, axis=0)
                X_test = X1_test if X_test is None else np.append(X_test, X1_test, axis=0)
                y_train = y1_train if y_train is None else np.append(y_train, y1_train, axis=0)
                y_test = y1_test if y_test is None else np.append(y_test, y1_test, axis=0)
                count += len(y)

    X_train = np.nan_to_num(X_train)
    X_train = np.clip(X_train, -1e10, 1e10)
    X_train = X_train.astype('float64')

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
