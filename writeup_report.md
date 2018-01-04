# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_12_29_16_05_26_995.jpg "Centered Driving"
[image2]: ./examples/left_2017_12_29_16_05_26_995.jpg "Left Camera"
[image3]: ./examples/right_2017_12_29_16_05_26_995.jpg "Right Camera"
[image4]: ./examples/center_2017_12_29_16_04_12_936.jpg "Normal Image"
[image5]: ./examples/center_2017_12_29_16_04_12_936_flip.jpg "Flipped Image"
[image6]: ./examples/center_2017_12_29_16_04_26_884_cropped.jpg "Cropped Image"
[image7]: ./examples/loss.png "MSE Loss"

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

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 57-61)

The model includes RELU layers to introduce nonlinearity (model.py lines 57-61), the data is normalized in the model using a Keras lambda layer (model.py line 55), and the input is cropped to eliminate unneccesary date (model.py line 56). Lastly the model uses fully connected layers of between 1 and 100 neurons. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used several laps of driving which included center lane driving, smooth turns, and recovery from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with a simple architecture and then increase complexity as needed.

I began by using a simple FC layer with one output neuron. I knew that this would likely not be sufficient since convolution layers are almost always used for image processing, but I wanted to start with a baseline.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a huge training and validation loss. This implied that the simple FC network was not a suitable architecture. 

I next updated the network to the LeNet5 architecture, since that is a relatively simple and powerful CNN. This brought a significant reduction in the training and validation loss. I trained this model and ran it in simulation. The vehicle was able to drive roughly straight, but would often wander off track occasionally.

To correct this wandering behavior, I used several augmentation and preprocessing techniques to improve the training data (see below). 

Lastly, I used the NVIDIA End-to-End CNN presented in the class. This drove down the training and validation loss by several orders of magnitude.

To combat the overfitting, I stopped the training after 5 epochs and used data augmentation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I used the model presented in Section 14 of the Behavioral Cloning module, [the NVIDIA End-to-End CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I selected this model because it has a real-world, demonstrable track record of robust autonomous driving. This model proved suitable to complete the track without any modifications to the kernaling, filters, or neurons. (model.py lines 54 - 66)


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Lambda         		| x / 255 - 0.5   								| 
| Cropping         		| 160x320x3 RGB image   						| 
| Convolution 5x5     	| 24 filters, 2x2 subsample						|
| RELU					|												|
| Convolution 5x5     	| 36 filters, 2x2 subsample						|
| RELU					|												|
| Convolution 5x5     	| 48 filters, 2x2 subsample						|
| RELU					|												|
| Convolution 3x3     	| 64 filters									|
| RELU					|												|
| Convolution 3x3     	| 64 filters									|
| RELU					|												|
| Fully connected		| 100 neurons 									|
| Fully connected		| 50 neurons 									|
| Fully connected		| 10 neurons 									|
| Fully connected		| 1 neurons 									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. This Here is an example image of center lane driving:

![alt text][image1]

I also utilized all three cameras from the driving simulator and added constant steering offsets to the left and right cameras. This provided the model examples of how to correct in the case of being uncentered, as well as tripling the sample data. Here is an example of the left and right cameras.

![alt text][image2]
![alt text][image3]

I augmented the data set by including mirrored data from all 3 cameras. This effectively doubled the simulation data on a "new" track. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 14158 number of data points. I then preprocessed this data by normalizing the image with a Lambda layer, and then cropping the top and bottom of the images to eliminated extraneous data. This allowed the model to focus on what was important. For example, here is a cropped image:

![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the following loss plot:

![alt text][image7]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
