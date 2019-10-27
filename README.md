---
### Contents

1. [Executive Summary](#executive-summary)
2. [Performance](#performance)
3. [Examples](#examples)
4. [Project Contribution](#project-contribution)
5. [User Guide](#user-guide)

### Executive Summary

This is TensorFlow Keras version of the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) model architecture for object detection.

This version of the SSD model uses MobileNet V2 as the base network to extract the fetaure maps for the first two predictor layers of the SSD network. The pretrained MobileNet weights are imported from the tensorflow.keras.applications package which were trained on the ImageNet dataset.

### Performance


### User Guide
#### Environment Setup
- 
#### Training
- Open train_ssd.py python script
- Modify variable data_dir to point to the data directory
- Run the python script