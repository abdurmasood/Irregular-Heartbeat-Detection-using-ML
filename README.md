# Irregular Heartbeat Detection using a Convolution Neural Network
Arrhythmia is an irregularity in the rate or rhythm of the heartbeat which, in some cases, may occur sporadically in a subjects daily life. Therefore, the automatic recognition of abnormal heartbeats from a large amount of ECG data is an important and essential task. In this project, a novel deep learning approach is proposed for ECG beat classification using a 2-Dimension convolution neural network **(Alexnet)**.  Experiments are done on a public dataset called the MIT-BIH Arrhythmia.

## Dependencies
Before you can run the scripts, there are a few dependencies that need to be downloaded using pip. These include:
- [Pandas](https://pypi.org/project/pandas/)
- [Numpy](https://pypi.org/project/numpy/)
- [WFDB](https://pypi.org/project/wfdb/)
- [Pillow](https://pypi.org/project/Pillow/)
- [Natsort](https://pypi.org/project/natsort/)
- [Keras](https://pypi.org/project/Keras/)
- [Tensorflow](https://pypi.org/project/tensorflow/)

## Setting up the Project
Before the project can be run, the MIT-BIH dataset needs to be downloaded so that the program scripts can access them. The pointers to the MIT-BIH dataset header files are available in this repository which can be downloaded using [**Git Large File Storage**](https://git-lfs.github.com/).

Git-LFS can be easily installed using the command `git lfs install`, and the header files can be downloaded using the command `git lfs fetch`.

## Directory Structure
```
├── README.md
├── mit-bih_database
│   ├── 100.csv
│   ├── 100annotations.txt
|        .
|        .
|        .
│   ├── 234.csv
│   └── 234annotations.txt
├── mit-bih_waveform
│   ├── 100.atr
│   ├── 100.dat
│   ├── 100.hea
|        .
|        .
|        .
│   ├── 234.atr
│   ├── 234.dat
│   └── 234.hea
└── scripts
    ├── _pycache_
    ├── cnn_model.py
    ├── directory_structure.py
    ├── extract_heartbeat.py
    ├── signal_api.py
    └── test.py
```

## Usage Instructions
Refer to the [Wiki](https://github.com/abdurmasood/Irregular-Heartbeat-Detection-using-ML/wiki) for instructions on how to train and run the model.
