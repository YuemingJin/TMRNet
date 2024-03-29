# Temporal Memory Relation Network for Workflow Recognition from Surgical Video
by [Yueming Jin](https://yuemingjin.github.io/), [Yonghao Long](https://scholar.google.com/citations?user=HIjQdFQAAAAJ&hl=zh-CN), [Cheng Chen](https://scholar.google.com.hk/citations?user=bRe3FlcAAAAJ&hl=en), [Zixu Zhao](https://scholar.google.com.hk/citations?user=GSQY0CEAAAAJ&hl=zh-CN), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

## Introduction
* The Pytorch implementation for our paper '[Temporal Memory Relation Network for Workflow Recognition from Surgical Video](https://arxiv.org/abs/2103.16327)', accepted at IEEE Transactions on Medical Imaging (TMI).

<p align="center">
  <img src="figure/overview_archi2.png"  width="800"/>
</p>

## Data Preparation
* We use the dataset [Cholec80](http://camma.u-strasbg.fr/datasets) and [M2CAI 2016 Challenge](http://camma.u-strasbg.fr/m2cai2016/index.php/program-challenge/).

* Training and test data split

   Cholec80: first 40 videos for training and the rest 40 videos for testing, following the original paper [EndoNet](https://arxiv.org/abs/1602.03012).

   M2CAI: 27 videos for training and 14 videos for testing, following the [challenge evaluation protocol](http://camma.u-strasbg.fr/m2cai2016/index.php/program-challenge/).

* Data Preprocessing: 
1. Using [FFmpeg](https://www.ffmpeg.org/download.html) to convert the videos to frames; 
2. Downsample 25fps to 1fps (Or can directly set the convert frequency number as 1 fps in the previous step); 
3. Cut the black margin existed in the frame using the function of ``change_size()`` in ``video2frame_cutmargin.py``;
```
Note: You also can directly use ``video2frame_cutmargin.py`` for step 1&3, you will obtain the cutted frames with original fps.
```
4. Resize original frame to the resolution of 250 * 250.

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


## Setup & Training

1. Check dependencies:
   ```
   - pytorch 1.0+
   - opencv-python
   - numpy
   - sklearn
   ```
2. Clone this repo
    ```shell
    git clone https://github.com/YuemingJin/TMRNet
    ```

2. Training model for building memory bank

* Switch folder ``$ cd ./code/Training memory bank model/``

* Run ``$ get_paths_labels.py`` to generate the files needed for the training

* Run ``$ train_singlenet_phase_1fc.py`` to start the training

3. Training TMRNet

* Switch folder ``$ cd ./code/Training TMRNet/``

* Put the well-trained model obtained from step 2 to folder ``./LFB/FBmodel/``

* Run ``$ get_paths_labels.py`` to generate the files needed for the training

* Set the args 'model_path' in ``train_*.py`` to ``./LFB/FBmodel/{your_model_name}.pth``

* Run ``$ train_*.py`` to start the training
  ```
  Note: In the first time to run train_*.py files, set the args 'load_LFB' to False to generate the memory bank
  We have three configurations about train_*.py:
  1.train_only_non-local_pretrained.py: only capture long-range temporal pattern (ResNet);
  2.train_non-local_mutiConv_resnet.py: capture long-range multi-scale temporal pattern (ResNet);
  3.train_non-local_mutiConv_resnest.py: capture long-range multi-scale temporal pattern (ResNeSt), achieving the best results.
  ```
## Testing

Our trained models can be downloaded from [Dropbox](https://www.dropbox.com/sh/4usgwrthboa3shq/AAC4S-fuQswq7usdPNq6q5yHa?dl=0).

* Switch folder ``$ cd ./code/eval/python/``
* Run ``$ get_paths_labels.py`` to generate the files needed for the testing
* Specify the feature bank path, model path and test file path in ./test_*.py
* Run ./test_*.py to generate results.
* Run ./export_phase_copy.py to export results as txt files.

We use the evaluation protocol of M2CAI challenge for evaluating our method.

* Switch folder ``$ cd ./code/eval/result/matlab-eval/``
* Run matlab files ./Main_*.m to evaluate and print the result.

## Citation
If this repository is useful for your research, please cite:
```
@ARTICLE{9389566,  
  author={Jin, Yueming and Long, Yonghao and Chen, Cheng and Zhao, Zixu and Dou, Qi and Heng, Pheng-Ann},  
  journal={IEEE Transactions on Medical Imaging},   
  title={Temporal Memory Relation Network for Workflow Recognition From Surgical Video},
  year={2021},  
  volume={40},  
  number={7},  
  pages={1911-1923},  
  doi={10.1109/TMI.2021.3069471}
}
```

### Questions

For further question about the code or paper, please contact 'ymjin5341@gmail.com'
