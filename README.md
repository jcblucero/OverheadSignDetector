# Introduction
This project is aimed at building an object detector, using Tensorflow and Python, to find and label highway overpass signs in an image.

![Three Sign Test](https://github.com/jcblucero/OverheadSignDetector/blob/master/doc/OverpassHeight_Three_Signs_Test_Out.PNG)

## Motivation
You may have seen news reports of trucks running into bridges that they do not fit under. There even is a site dedicated to one specific low bridge http://11foot8.com/.

I thought it would be interesting to come up with a solution that could find bridge overpass height signs and identify the height of the bridge. Then, if the vehicle height is known, the system could notify the driver that they are about to crash into a bridge that is too low for the vehicle. 

## Setup
I ran everything on Windows 10, but the same setup should work on Linux.
Download the v1.13 release of the Tensorflow Models directory: https://github.com/tensorflow/models/tree/r1.13.0

Then download this repository into the models/research/object_detection folder.

## Installing Tensorflow

I followed this guide for Installing Tensorflow (and for training): https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

I used the following software versions:

Software | Version
------------ | -------------
Tensorflow | 1.15.0
Python | 3.5.6
CUDA | 10.0
CUDNN | 7.6.5.32
TensorRT | 6.0.1.5

## Running
Run Overhead_Sign_Detection.py from the models/research/object_detection folder.
You can see a sample image input in the main function of the script.

## Results
![Two Sign](https://github.com/jcblucero/OverheadSignDetector/blob/master/doc/OverpassHeight_15_0_And_15_4_2.PNG) 
![Test Image](https://github.com/jcblucero/OverheadSignDetector/blob/master/doc/OverpassHeight_16_5_3.PNG)
[![Youtube Link](https://github.com/jcblucero/OverheadSignDetector/blob/master/doc/VideoImage.PNG)](http://www.youtube.com/watch?v=k1BO4s32JA8)


