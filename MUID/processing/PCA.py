import numpy as np


def mean_data(data):
    return


def pca(XMat, k):
    average = np.mean(XMat, axis=0)
    m, n = np.shape(XMat)
    data_adjust = (XMat - average) / np.std(XMat, axis=0)

    covX = np.cov(data_adjust, rowvar=False)

    featValue, featVec = np.linalg.eigh(covX)

    index = np.argsort(featValue)
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.array(featVec[:, index[:-(k + 1):-1]])
        finalData = np.dot(data_adjust, selectVec)
    return finalData
