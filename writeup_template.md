# **Behavioral Cloning - Udacity Self-Driving Car Engineer Nanodegree** 



### In this project I am using Deep learning and convultional neural networks to teach the computer to drive a car autonomously by cloning driving behaviour while driving in a simulator. I am using the data provided by Udacity.

The data collected from the simulator contains images captured by 3 cameras (left, right and center) on the dashboard, and a data.csv file which has the mappings of center, left and right images and the corresponding steering angle, throttle, brake and speed.

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/Architecture.png "Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Loading data

Using the data set provided by Udacity I did the following as preprocessing

* Converted the images from BGR to RGB.
* Flipped every image and appended to the data set to augment the data and also to set the steering angle (Since the simulator track has mostly left side steering angles, flipping the images add right side steering angle).
* Added a correction of 0.2 to steering angle for left images and -0.2 to the right images.
* Shuffled the data.
* Divided the data set for training (85%) and validation(15%) set.

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48, then 3x3 filter sizes and depths 64.

The model includes ELU layers, and the data is normalized in the model using a Keras lambda layer. 

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 85). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).
* No of epochs= 5
* Optimizer Used- Adam
* Learning Rate- Default 0.001
* Validation Data split- 85 - 15
* batch size= 32
* Correction factor for steering angle - 0.2
