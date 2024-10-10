import numpy as np
import scipy
import pylab
import pickle
from processing import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Track import Track, MeasurePoint
from Hypothesis import *
from scipy import signal
from scipy.fftpack import fft, ifft

lumbuda = 2 * 2.727e-2
d = 0.03  # distance between adjacent antennas in the linear antenna array
center_frequency = 5.5e9
Ts = 1e-3
sample_rate = 1000
Twin = 0.4
tx = 1
rx = 7  # number of rx antennas
sub_carr_num = 30
c = 3e8  # speed of light

K = int(np.floor(rx / 2) + 1)
L = int(np.floor(sub_carr_num / 2))
T = 1


def butterWorth_lowpass(csimat, factor):
    b, a = signal.butter(5, 2 * factor / (1000), 'lowpass')  # v=1m/s f=2*v/lumbuda wn=2f/fs[2,25]
    noise_remove = signal.filtfilt(b, a, csimat)
    return noise_remove


def complexToLatitude(subcarrier):
    return np.round(abs(subcarrier), 4)


def complexToPower(subcarrier):
    return np.round(pow(abs(subcarrier), 2), 4)


def get_amp(pair):
    amp = None
    for subcarrier in pair:
        amp = np.array([complexToLatitude(subcarrier)]) if amp is None else np.append(
            amp, [complexToLatitude(subcarrier)], axis=0)
    return amp


def pca_CSI_Matrix(CSI_Matrix, num_components):
    CSI_Matrix = np.transpose(CSI_Matrix, [0, 2, 1])
    CSI_pca = None
    for pair in CSI_Matrix:  # sub PCA and denoise
        PCA_csi = PCA.pca(pair, num_components)  # (5000,5)
        PCA_csi = np.sum(PCA.pca(pair, num_components), axis=1)
        # PCA_power = dwtfilter(PCA_power).filterOperation()
        # De_PCA_power2 = butterWorth_bandpass(PCA_power, 1)
        CSI_pca = np.array([PCA_csi]) if CSI_pca is None else np.append(
            CSI_pca, [PCA_csi], axis=0)
    # CSI_power_pro=np.transpose(CSI_power_pro)
    print('CSI_pca shape is ', CSI_pca.shape)

    return CSI_pca


def avg_CSI_Matrix(CSI_Matrix):
    CSI_avg = None
    for pair in CSI_Matrix:  # sub PCA and denoise
        avg_csi = np.mean(pair, axis=0)

        CSI_avg = np.array([avg_csi]) if CSI_avg is None else np.append(
            CSI_avg, [avg_csi], axis=0)
    return CSI_avg


def calculate_cmov(csi_pca):
    conjugate_product = np.multiply(csi_pca, np.conjugate(csi_pca))
    power_spectrum = np.abs(conjugate_product)
    average_spectrum = np.mean(power_spectrum, axis=1)
    cavg_matrix = np.tile(average_spectrum[:, np.newaxis], (1, power_spectrum.shape[1]))
    cmov_matrix = power_spectrum - cavg_matrix
    return cmov_matrix


def findPeak2D(spectrum, sourceNum):
    max_value = np.max(spectrum)
    peakIndexes_before, peakValues_before = [], []
    for i in range(0, len(spectrum) - 1):
        if i == 0:
            for j in range(1, len(spectrum[0]) - 1):
                if (spectrum[i, j - 1] <= spectrum[i, j] >= spectrum[i, j + 1]) \
                        or (spectrum[i, j - 1] <= spectrum[i, j] >= spectrum[i, j + 1]):
                    peakIndexes_before.append([i, j])
                    peakValues_before.append(spectrum[i, j])
        else:
            for j in range(1, len(spectrum[0]) - 1):
                if (spectrum[i, j - 1] <= spectrum[i, j] > spectrum[i, j + 1] and spectrum[i - 1, j] < spectrum[i, j] >
                    spectrum[i + 1, j]) \
                        or (spectrum[i, j - 1] < spectrum[i, j] >= spectrum[i, j + 1] and spectrum[i - 1, j] < spectrum[
                    i, j] > spectrum[i + 1, j]) \
                        or (spectrum[i, j - 1] < spectrum[i, j] > spectrum[i, j + 1] and spectrum[i - 1, j] <= spectrum[
                    i, j] > spectrum[i + 1, j]) \
                        or (spectrum[i, j - 1] < spectrum[i, j] > spectrum[i, j + 1] and spectrum[i - 1, j] < spectrum[
                    i, j] >= spectrum[i + 1, j]):
                    peakIndexes_before.append([i, j])
                    peakValues_before.append(spectrum[i, j])

    peakIndexes, peakValues = [], []
    for i in range(len(peakValues_before)):
        #     if peakValues_before[i] >= max_value * 0.6:
        peakIndexes.append(peakIndexes_before[i])
        peakValues.append(peakValues_before[i])

    if len(peakIndexes) < sourceNum:
        return peakIndexes
    else:
        Z = zip(peakValues, peakIndexes)
        Zipped = sorted(Z, reverse=True)  # descend order
        valueDescend, indexDescend = zip(*Zipped)  # in type tuple
        selectIndex = list(indexDescend)[0: sourceNum]
        return selectIndex


def fft2_aoa(CSI_mov, window_len):
    N = 7
    AoA_grid_gap = lumbuda / (6 * d)  # 1/(7*d)
    V_grid_gap = lumbuda / (window_len / sample_rate)  # FS/N
    spectrum = np.fft.fft2(CSI_mov, (30, 400))
    freql = np.fft.fftfreq(np.size(spectrum, 0), lumbuda / 2)
    freqt = np.fft.fftfreq(np.size(spectrum, 1), 0.001)
    specshift_abs = np.abs(spectrum)
    specshift_abs = np.transpose(specshift_abs)

    result = findPeak2D(specshift_abs, 2)
    AoA = []
    for i in range(len(result)):
        aoa_consider = (result[i][1]) * AoA_grid_gap - 0.33
        V_consider = (result[i][0]) * V_grid_gap

        print('the ', i + 1, 'th peak   ', aoa_consider, V_consider)
        print('the ', i + 1, 'th peak-- ', np.arccos(0.67 - aoa_consider) / 3.14 * 180 - 48, V_consider)
        AoA.append(-(np.arccos(0.67 - aoa_consider) / 3.14 * 180 - 48))

    return AoA


def get_AoA_list(CSI_mov, Twin=0.4, packet_start=0):
    AoA_list = []
    fileLen = CSI_mov.shape[-1]
    window_start = 0
    slide_window_len = int(sample_rate * Twin)  # 1000*0.4=400
    step_len = 200

    while window_start + slide_window_len <= fileLen:
        print('test package start is ', window_start + packet_start)
        AoA = fft2_aoa(CSI_mov[:, window_start: window_start + slide_window_len], slide_window_len)
        # res = np.log(np.abs(fshift))
        # plt.imshow(res, 'gray'), plt.title('Fourier Image')
        # plt.show()
        AoA_list.append(AoA)
        window_start += step_len

    return AoA_list


def get_AoA_list_v2(CSI_mat, Twin=0.4, packet_start=0):
    AoA_list = []
    fileLen = CSI_mat.shape[-1]
    window_start = 0
    slide_window_len = int(sample_rate * Twin)
    step_len = 200

    while window_start + slide_window_len <= fileLen:
        CSI_mov = calculate_cmov(CSI_mat[:, :, window_start: window_start + slide_window_len])
        CSI_mov = avg_CSI_Matrix(CSI_mov)
        CSI_mov = butterWorth_lowpass(CSI_mov, 40)
        AoA = fft2_aoa(CSI_mov, slide_window_len)
        AoA_list.append(AoA)
        window_start += step_len

    return AoA_list


def AoA_track(AoA_list):
    T = 0.5
    pd = 0.95
    betaFT = 0.7
    betaNT = 0.001
    V = 180
    maxTracks = 7

    def ExpandHypothesis(hyp, hypothesisMat):
        """
        Hypothesis which include a track list and the probability of this hypothesis.

        Parameters
        -------------
        hyp: class Hypothesis
            The origin hypothesis

        hypothesisMat: list (N*M)
            Hypothesis matrix generated by hyp.CalHypothesisMat
            where there are M measurements and N possible hypotheses.

        Returns
        ----------
        ret: list class Hypothesis
            Expanded hypotheses.
        """
        ret = []
        for situation in hypothesisMat:
            newHyp = copy.deepcopy(hyp)
            prob = 1
            nngt = len(hyp.tracks)
            ndt, nft, nnt = 0, 0, 0
            visParent = set()
            for parent, measure in zip(situation, measures):
                if parent == 0:  
                    nft += 1
                    prob *= 1 / V
                elif parent <= nngt: 
                    ndt += 1
                    prob *= newHyp.tracks[parent - 1].estimateP(MeasurePoint(measure))  # N(z-Hx_;0,H.P_.H^T+R)
                    # print('P=',newHyp.tracks[parent-1].estimateP(MeasurePoint(measure[0])))
                    newHyp.tracks[parent - 1].AddPoint(MeasurePoint(measure))  # update P ,并添加到trackpoint里
                    visParent.add(parent - 1)
                else: 
                    nnt += 1
                    prob *= 1 / V
                    newHyp.tracks.append(newTracks[parent - len(hyp.tracks) - 1])
                    visParent.add(len(newHyp.tracks) - 1)
            assert nngt >= ndt
            newHyp.prob = pow(pd, ndt) * pow(1 - pd, nngt - ndt) * pow(betaFT, nft) * pow(betaNT, nnt) * prob * hyp.prob

           
            # print('hyp.prob={0},newHyp.prob={1},prob={2}'.format(hyp.prob,newHyp.prob,prob))
            # for i,track in enumerate(newHyp.tracks):
            #     if i not in visParent:
            #         track.AddPoint(track.PredictNextPos())
            #         track.fail+=1
            # for i in reversed(range(len(newHyp.tracks))):
            #     if newHyp.tracks[i].fail>=maxFail:
            #         newHyp.tracks.pop(i)
            ret.append(newHyp)

        return ret

    data = AoA_list.copy()  # [0:40]
    SAVE_FIG = None
    SILENCE = None
    ####### variables #######
    idTrack = 1
    hyps = [Hypothesis()]

    ####### plot results #######
    x, y, c = [], [], []
    xori, yori = [], []

    for i, measures in enumerate(data):
        ####### generate new tracks using measures #######
        if not SILENCE:
            print('---------')
            print('Frame{}:'.format(i))
        newTracks = []
        for measure in measures:  #
            newTracks.append(Track(MeasurePoint(measure), T, idTrack))
            xori.append(i)
            yori.append(measure)
            idTrack += 1

        if not SILENCE:
            print('measures:', measures)

        ####### generate hypotheses tree #######
        newHyps = []
        for hyp in hyps:
            hyp.Predict()
            # print('hyp.prob=',hyp.prob)
            hypothesisMat = hyp.CalHypothesisMat(newTracks)
            # print('hypothesisMat=',hypothesisMat)
            expandedHyps = ExpandHypothesis(hyp, hypothesisMat)

            newHyps.extend(expandedHyps)

        ####### merge and prune hypotheses #######
        PruneHypsWithMaxTracks(newHyps, maxTracks)
        if len(newHyps) > 50:
            SortHypothesis(newHyps)
            newHyps = MergeHyps(newHyps)
            SortHypothesis(newHyps)
            newHyps = PruneHyps(newHyps, 10)

        ####### plot data #######
        tmp = np.array([item.prob for item in newHyps])
        maxval = np.max(tmp)
        idx = np.where(tmp == maxval)[0]
        # print(tmp)
        # print(idx,[len(newHyps[i].tracks) for i in idx])
        if not SILENCE:
            print('tracks:')
        for j in idx:
            for item in newHyps[j].tracks:
                if not SILENCE:
                    print(item.idTrack, end=' ')
                    print(item.trackPoint[-1])
                x.append(i)
                y.append(item.trackPoint[-1].theta)
                c.append(item.idTrack)

        ####### normalize hypotheses weight #######
        NormalizeWeight(newHyps)

        hyps = newHyps

    '''calculte track nums and remove exceeding tracks'''
    track_num = []
    [track_num.append(x) for x in c if x not in track_num]
    track_num = sorted(track_num)
    track_id_count = []
    for i in range(len(track_num)):
        track_id_count.append(c.count(track_num[i]))

    exceeding_num = len(track_num) - 3
    print('exceeding_num', exceeding_num)
    if exceeding_num > 0:
        for i in range(exceeding_num):
            index = track_id_count.index(min(track_id_count))
            track_id_count.pop(index)
            track_num.pop(index)
    print('track_num and track_id_count after process: ', track_num, track_id_count)
    AoA_Matrix = np.zeros((len(track_num), len(AoA_list)))
    for i in range(len(c)):
        if track_num.count(c[i]) > 0:
            index = track_num.index(c[i])
            AoA_Matrix[index][x[i]] = round(y[i], 2)
    for i in range(len(AoA_Matrix)):
        for j in range(len(AoA_Matrix[0]) - 2, -1, -1):
            if AoA_Matrix[i][j] == 0:
                AoA_Matrix[i][j] = AoA_Matrix[i][j + 1]

    '''sort AoA_Matrix from min abs to max abs using first column'''
    '''through this, the tracking AoA in one row can correspond to the fixed person'''
    AoA_col = -np.sort(-AoA_Matrix[:, 0])
    AoA_Matrix_aline = np.zeros(AoA_Matrix.shape)
    for i in range(len(AoA_col)):
        index = np.where(AoA_Matrix[:, 0] == AoA_col[i])[0]
        AoA_Matrix_aline[i] = AoA_Matrix[index]
    print(AoA_Matrix_aline[:, 0])

    return AoA_Matrix_aline


if __name__ == '__main__':
    filename = 'E:/Programs/PycharmProjects/multi_ac_monitor/data_10_24_2p/sample_rate_1000/comb1/multi_0/1CSIcutData_aline_10as/raw_csi_target_1.pkl'
    with open(filename, 'rb') as f:
        CSIMatrix = pickle.load(f)
    CSIMatrix = CSIMatrix[0]
    packet_start = 0
    packet_end = 5000
    CSIMatrix = CSIMatrix[:, :, packet_start:packet_end]
    CSIMatrix[2] = CSIMatrix[2] * 3
    CSIMatrix = np.delete(CSIMatrix, [3, 6], axis=0)  # (7*30*packet)

    aoa_list = get_AoA_list_v2(CSIMatrix, Twin, packet_start)
    aoa_tracking_mat = AoA_track(aoa_list)


