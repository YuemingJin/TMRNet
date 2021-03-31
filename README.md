# Temporal Memory Relation Network for Workflow Recognition from Surgical Video
by [Yueming Jin](https://yuemingjin.github.io/),[Yonghao Long](https://scholar.google.com/citations?user=HIjQdFQAAAAJ&hl=zh-CN), [Cheng Chen](https://scholar.google.com.hk/citations?user=bRe3FlcAAAAJ&hl=en), [Zixu Zhao](https://scholar.google.com.hk/citations?user=GSQY0CEAAAAJ&hl=zh-CN), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction
* The Pytorch implementation for our TMI 2021 paper '[Temporal Memory Relation Network for Workflow Recognition from Surgical Video](https://ieeexplore.ieee.org/document/9389566)'. 

<p align="center">
  <img src="figure/overview_archi2.png"  width="800"/>
</p>

<!-- * The Code contains two parts: motion learning (flow prediction and flow compensation) and semi-supervised segmentation. -->

### Data Preparation
* We use the dataset [Cholec80](http://camma.u-strasbg.fr/datasets).

* Please follow the data preprocessing steps in this [repository](https://github.com/keyuncheng/MF-TAPNet).

* The structure of data folder is arranged as follows:
```
(root folder)
├── data
|  ├── cholec80
|  |  ├── cutMargin
|  |  |  ├── 1
|  |  |  ├── 2
|  |  |  ├── 3
|  |  |  ├── ......
|  |  |  ├── 80
|  |  ├── phase_annotations
|  |  |  ├── video01-phase.txt
|  |  |  ├── ......
|  |  |  ├── video80-phase.txt
├── code
|  ├── ......

```


### Setup & Usage for the Code

1. Check dependencies:
```
- pytorch 1.0+
- opencv-python
- numpy
- sklearn
```

2. Building long-range memory bank
Go into folder ``code/Building long-range memory bank``


2. Data preloading:
* Run ``$ get_paths_labels.py`` to generate the files needed for the .

3. Semi-supervised segmentation (./segmentation/):
```
$ bash train.sh
```
Note: You may try other models from /Models/plane_model.py

### Citation
If this repository is useful for your research, please cite:
```
@inproceedings{zhao2020learning,
  title={Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video},
  author={Zhao, Zixu and Jin, Yueming and Gao, Xiaojie and Dou, Qi and Heng, Pheng-Ann},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={679--689},
  year={2020},
  organization={Springer}
}
```

### Questions

For further question about the code or paper, please contact 'ymjin@cse.cuhk.edu.hk'