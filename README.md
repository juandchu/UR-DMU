# Masters Thesis Project JD Chu
This repo contains the implementation of a model for anomaly detection specifically designed for drone-recorded video data in cargo/port areas. It is based on the architecture of: 

> **Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection (UR-DMU)**
> 
> Hang Zhou, Junqing Yu, Wei Yang


## Training

### Setup
**To train this model it is necessary to extract the feature embedding of the videos using:**
1. `video2frame_split.py` $\rightarrow$ Specify folder with mp4 videos for frame extraction
2. 






-----------------------
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
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)

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
