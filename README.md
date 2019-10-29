---
### Contents

1. [Executive Summary](#executive-summary)
2. [Performance](#performance)
3. [Project Contribution](#project-contribution)
4. [User Guide](#user-guide)

### Executive Summary

This is TensorFlow Keras version of the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) model architecture for object detection.

This version of the SSD model uses MobileNet V2 as the base network to extract the fetaure maps for the first two predictor layers of the SSD network. The pretrained MobileNet weights are imported from the tensorflow.keras.applications package which were trained on the ImageNet dataset.

### Performance

### Project Contribution

### User Guide
#### Environment Setup
conda create -n rtav python=3.6 numpy=1.16.5 opencv=4.1.0 matplotlib tensorflow=1.13.1 tensorflow-gpu=1.13.1 cudatoolkit=9.0 cudnn=7.1.4 scipy=1.1.0 scikit-learn=0.21.3 pillow=5.1.0 spyder=3.3.2 cython=0.29.2 pathlib=1.0.1 ipython=7.2.0 imutils=0.5.2 yaml pandas keras keras-gpu pydot graphviz scikit-image imgaug librosa
#### Training
- Open train_ssd.py python script
- Modify variable data_dir to point to the data directory
- Run the python script