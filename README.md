# [Master's Thesis] Drone-Based Video Anomaly Detection in Port Environments
### Implementation Based on UR-DMU (Uncertainty Regulation with Dual Memory Units)

This repository contains the implementation of a specialized Video Anomaly Detection (VAD) framework designed for **drone-recorded footage** in cargo and port facilities. 

The project adapts the architecture introduced in:

> **Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection (UR-DMU)**
> *Authors: Hang Zhou, Junqing Yu, Wei Yang (AAAI 2023)*

---
### Dataset Structure
For a dataset of n videos:
```text
data/
├── train/
│   ├── normal/
│   │   ├── video_001/video_001.mp4
│   │   ├── video_002/video_002.mp4
│   │   └── ... (additional normal videos)
│   └── abnormal/
│       ├── video_001/video_001.mp4
│       ├── video_002/video_002.mp4
│       └── ... (additional abnormal videos)
└── test/
    ├── video_test_01/
    │   ├── video_test_01.mp4
    │   └── labels.csv      # Ground truth (start/end frames)
    ├── video_test_02/
    │   ├── video_test_02.mp4
    │   └── labels.csv
    └── ...
```
**Labels.csv** should be set as: 

| start | end  | label |
|-------|------|-------|
| 100   | 800  | 1     |
| 950   | 1500 | 1     |
| 2000  | 2500 | 1     |

---


### Pre Training Pipeline
**To train the main model it is necessary to extract the feature embedding of the videos in the dataset.**

1.<u>Extract the frames of each video</u>

`video2frame_split.py` $\rightarrow$ If the data is set as the previous instructions, this will generate a new folder with the same folder structure but the video frames.  

2. 






<!-- -----------------------
> [**XD-Violence 5-crop I3D features**](https://roc-ng.github.io/XD-Violence/)
> 
> [**best performance ckpt for UCF**](models/ucf_trans_2022.pkl)
>
> [**best performance ckpt for XD**](models/xd_trans_2022.pkl)

You can also use the I3D model to extract features from [**preprocess**](feature_extract/README.md).

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/XD_Train.list` and `list/XD_Test.list`. 
- Feel free to change the hyperparameters in `option.py`
### Train and test the UR-DMU
After the setup, simply run the following command: 

start the visdom for visualizing the training phase

```
python -m visdom.server -p "port"(we use 2022)
```
Traing and infer for XD dataset
```
python xd_main.py
python xd_infer.py
```
Traing and infer for UCFC dataset
```
python ucf_main.py
python ucf_infer.py
```

## References
We referenced the repos below for the code.

* [RTFM](https://github.com/tianyu0207/RTFM)
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet) -->

## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{URDMU_zh,
  title={Dual memory units with uncertainty regulation for weakly supervised video anomaly detection},
  author={Zhou, Hang and Yu, Junqing and Yang, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={3},
  pages={3769--3777},
  year={2023}
}
```
---
