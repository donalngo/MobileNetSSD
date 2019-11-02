---
### Contents

1. [Executive Summary](#executive-summary)
2. [User Guide](#user-guide)

### Executive Summary



This repository explores the methodology behind the popular lightweight object detector MobileNet SSD. MobileNet SSD is actually a combination of 2 separate algorithms published in separate papers. The MobileNet is a lightweight feature extractor meant for deployment in Mobile phones and has gone through a couple of iterations. The [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) on the other hand is an innovative object detector model that utilizes multiple feature maps to perform object detection. The original SSD was in fact trained on VGG and not MobileNet. 
In this project we explore the use of MobileNet v2 combined with SSD to do object detections of everyday supermarket items. Manpower shortage together with an ageing working population are problems that many companies around the world are facing today. The problem of finding enough manpower to operate the increasing number of 24-hours supermarkets and increasing probability of payment errors at checkout due to the ageing population of cashiers are definitely problems that the supermarkets face today. 
Our team hopes to solve both problems through the use of computer vision. An automated checkout Point of Sale system that utilizes a lightweight object detection algorithm. As such our team has gathered a collection of simple everyday items such as bananas and apples to train our object detector classifier.  
The team faced numerous challenges and went through multiple iterations in not just the model creation process but also the data collection and pre-processing. 
The team managed to obtain a validation loss of 0.43871.

This is a TensorFlow Keras implementation of the  model architecture discussed above.



### User Guide
#### Environment Setup
conda create -n rtav python=3.6 numpy=1.16.5 opencv=4.1.0 matplotlib tensorflow=1.13.1 tensorflow-gpu=1.13.1 cudatoolkit=9.0 cudnn=7.1.4 scipy=1.1.0 scikit-learn=0.21.3 pillow=5.1.0 spyder=3.3.2 cython=0.29.2 pathlib=1.0.1 ipython=7.2.0 imutils=0.5.2 yaml pandas keras keras-gpu pydot graphviz scikit-image imgaug librosa


#### Training
- Open train_ssd.ipynb python notebook
- Modify variable data_dir to point to the data directory
- Run all the cells to train the model and plot predictions in the last cell