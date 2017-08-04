# Spatial-Temporal-Pooling-Networks-ReID
Code for our ICCV 2017 paper -- Jointly Attentive Spatial-Temporal Pooling Networks for Video-based Person Re-Identification

If you use this code please cite:

```
@inproceedings{shuangjiejointly,
  	title={Jointly Attentive Spatial-Temporal Pooling Networks for Video-based Person Re-Identification},
  	author={Shuangjie Xu, Yu Cheng, Kang Gu, Yang Yang, Shiyu Chang and Pan Zhou},
  	booktitle={ICCV},
  	year={2017}
}
```

## Dependencies
The following libaries are necessary:
* [torch](http://torch.ch/) and its package (nn, nnx, optim, cunn, cutorch, image, rnn , inn). [Installation guide](http://torch.ch/docs/getting-started.html#_)
* Matlab for data preparation

## Data Preparation
Download and extract datasets [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html), [PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11) and [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) into the ```data/``` directory. ```data/iLIDS-VID``` for example.

Modify and run ```data/computeOpticalFlow.m``` with Matlab to generate Optical Flow data. Optical Flow data will be generated in the same dir of your datasets. ```data/iLIDS-VID-OF-HVP``` for example.

MARS needs some extra codes to randomly choose two videos for a person (cam1 and cam2). Will release soon.

## Training
