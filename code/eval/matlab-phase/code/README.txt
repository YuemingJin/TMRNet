This folder includes: 
- m2cai16-workflow training data:
  - 27 cholecystectomy videos
  - 27 phase annotations
  - 27 corresponding timestamped annotations for visualization
- m2cai16-workflow testing data:
  - 14 cholecystectomy videos
  - 14 phase annotations
  - 14 corresponding timestamped annotations for visualization
- MATLAB scripts to perform the evaluation
- a README file

================================================
DATASET DESCRIPTION
================================================

The m2cai16-workflow dataset contains 41 videos of cholecystectomy procedures from University Hospital of Strasbourg/IRCAD (Strasbourg, France) and Hospital Klinikum Rechts der Isar (Munich, Germany). The dataset is split into two parts: training subset (containing 27 videos) and testing subset (14 videos). For the sake of fairness, please do not use any of the testing videos during your training process.

The videos are recorded at 25 fps. All the frames are fully annotated with 8 defined phases: (1) trocar placement, (2) preparation, (3) calot triangle dissection, (4) clipping and cutting, (5) gallbladder dissection, (6) galbladder packaging, (7) cleaning and coagulation, and (8) gallbladder retraction.

The annotation file contains a table, consisting of 2 columns. Each row contains an annotation for an image in the video (except for the header of the table). 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. The second column is the phase label for the corresponding frame.

The timestamped annotations are provided to facilitate the visualization of the phases. These timestamped annotations can be visualized as subtitles on video players. This method has been successfully tested on VLC media player, by dragging the timestamped annotation to the player while the video is playing.

================================================
EVALUATION SCRIPT
================================================

The main script is Main.m. It computes the jaccard index for each phase and the accuracy over the videos. The evaluation is relaxed on the boundaries of the phases with a 10-second window.

The script assumes that the annotation files and the result files are in the same folder. It also assumes the naming formats of the annotation and the result files are workflow_video_<#ID>.txt and the workflow_video_<#ID>_pred.txt, respectively. The content of the result files should be in the same format as the format in the annotation files. 

We have included two result file examples in the folder: (workflow_video_02_pred.txt and workflow_video_04_pred.txt). These examples are generated with random values.

================================================
LICENSE AND REFERENCES
================================================

This dataset could only be generated thanks to the continuous support from the surgeons from both partner hospitals. In order to properly credit them for their efforts, you are kindly requested to cite the two papers that led to the generation of this dataset:
- A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Trans. on Medical Imaging 2016.
- D. Ostler, M. Kranzfelder, R. Stauder, D. Wilhelm, H. Feussner, A. Schneider. A centralized data acquisition framework for operating theatres. In IEEE 17th International Conference on e-Health Networking 2015.

The m2cai16-workflow dataset is publicly released under the Creative Commons licence CC-BY-NC-SA 4.0. This implies that:
- the dataset cannot be used for commercial purposes,
- the dataset can be transformed (additional annotations, etc.),
- the dataset can be redistributed as long as it is redistributed under the same license with the obligation to cite the contributing work which led to the generation of the m2cai16-workflow dataset (mentioned above).

By downloading and using this dataset, you agree on these terms and conditions.

================================================
CONTACT
================================================

The official webpage of the challenge can be found here: http://camma.u-strasbg.fr/m2cai2016
To see a list of recognition results on this dataset, please visit our web page: http://camma.u-strasbg.fr/result-list-m2cai16-workflow

Any questions regarding the dataset or the challenge can be sent to: m2cai2016@gmail.com
