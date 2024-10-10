# MUID
This repository contains the experiments done in the work: "Wang L, Yang X, Zhou S, et al. MUID: Multi-person Gait Identification with Commodity Wi-Fi."

The part of train and test data can be download in [https://drive.google.com/drive/folders/1IIdgjjJgNNk3DWh6Q30SqmAxXK-zW-vp?usp=drive_link](https://drive.google.com/drive/folders/1IIdgjjJgNNk3DWh6Q30SqmAxXK-zW-vp?usp=drive_link). And the rest of data can be download in [https://pan.baidu.com/s/1ve_xykl-gRpE_YXmVKKhsQ?pwd=n60x](https://pan.baidu.com/s/1ve_xykl-gRpE_YXmVKKhsQ?pwd=n60x)

The code for the signal processing section, including ACF-based speed extraction, Doppler spectrum calibration and TAC information extraction, is placed in the file of Processing.

The code for the two-branch feature fusion identification network is placed in the file of Identification. You can run it to show the overall identification performance in 2-person and 3-person cases.

In addition, we also put the folders: Baseline, including the multi-person identification method (magauth) and the single-person identification methods (wifiu and gaitway). 
