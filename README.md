# Temporal Memory Relation Network for Workflow Recognition from Surgical Video
by [Yueming Jin](https://yuemingjin.github.io/), [Yonghao Long](https://scholar.google.com/citations?user=HIjQdFQAAAAJ&hl=zh-CN), [Cheng Chen](https://scholar.google.com.hk/citations?user=bRe3FlcAAAAJ&hl=en), [Zixu Zhao](https://scholar.google.com.hk/citations?user=GSQY0CEAAAAJ&hl=zh-CN), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction
* The Pytorch implementation for our TMI 2021 paper '[Temporal Memory Relation Network for Workflow Recognition from Surgical Video](https://arxiv.org/abs/2103.16327)'. 

<p align="center">
  <img src="figure/overview_archi2.png"  width="800"/>
</p>

<!-- * The Code contains two parts: motion learning (flow prediction and flow compensation) and semi-supervised segmentation. -->

### Data Preparation
* We use the dataset [Cholec80](http://camma.u-strasbg.fr/datasets) and [M2CAI 2016 Challenge](http://camma.u-strasbg.fr/datasets).

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

2. Training memory bank model

* Switch folder ``$ cd ./code/Training memory bank model/``

* Run ``$ get_paths_labels.py`` to generate the files needed for the training.

* Run ``$ train_singlenet_phase_1fc.py`` to start the training.

3. Training TMRNet

* Switch folder ``$ cd ./code/Training TMRNet/``

* Put the model obtained from step 2 to folder ``./LFB/FBmodel/``.

* Run ``$ get_paths_labels.py`` to generate the files needed for the training.

* Set the args 'model_path' in ``train_*.py`` to ``./LFB/FBmodel/{your_model_name}.pth``

* (The first time to run train files) Set the args 'load_LFB' to False to generate the memory bank

* Run ``$ train_*.py`` to start the training.


### Citation
If this repository is useful for your research, please cite:
```
@ARTICLE{9389566,
    author={Y. {Jin} and Y. {Long} and C. {Chen} and Z. {Zhao} and Q. {Dou} and P. -A. {Heng}},
    journal={IEEE Transactions on Medical Imaging}, 
    title={Temporal Memory Relation Network for Workflow Recognition from Surgical Video}, 
    year={2021},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TMI.2021.3069471}
}
```

### Questions

For further question about the code or paper, please contact 'ymjin@cse.cuhk.edu.hk'
