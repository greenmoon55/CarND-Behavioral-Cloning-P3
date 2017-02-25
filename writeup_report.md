#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1.png "Original Steering"
[image2]: ./examples/2.png "Data from three cameras"
[image3]: ./examples/3.png "Data"
[image4]: ./examples/4.png "Final data"
[image5]: ./examples/5.png "Data in training"
[image6]: ./examples/6.png "Validation loss"
[image7]: ./examples/7.png "New validation loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 recorded on track 1

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used a lambda layer to normalize the data to a range of -0.5 to 0.5. My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 237-246) 

The model includes RELU layers after each convolution layer to introduce nonlinearity. Then I used flatten layer and four dense layers of size 80, 40, 16 and 1.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after each dense layer in order to reduce overfitting. L2 regularization is also to prevent overfitting.

The model was trained and validated on Udacity Dataset. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 256).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of random brightness, flipping, multiple cameras, reducing data of driving straight to create a balanced dataset.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was trail and error.

My first step was to use a simple convolution neural network model of two convolution layers. I thought this model might be appropriate because someone posted that it was enough.


But I found the data was imbalanced and the car in the simulator was driving in a straight line.

![imbalanced data][1.png]

I tried to use images from left and right camera with a correction value added to steering values.


![imbalanced data][2.png]

I limited number of data between 0.0 and 0.1 to 500. 

![imbalanced data][3.png]

Lastly, If the steering value is larger than 0.15, I made steering value larger by using left or right camera. the steering value is larger than 0.2, I added the modified data twice to further combat the imbalance problem. 

![imbalanced data][4.png]

I also used random brightness, flipping the image in the generator for data augmentation. The following picture showed all data used in the training process.

![imbalanced data][5.png]

In order to gauge how well the model was working, I split my image and steering angle data into a training and 10% validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set and the mse did not drop significantly after each epochs. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layers and used l2 regularization. The car could drive but could not turn well and fell off in some corners. The validation loss is still quite high.

![validation loss][6.png]

Then I thought my neural network with two convolutional layers was too simple, so I started an AWS instance and used an architecture from https://github.com/ancabilloni/SDC-P3-BehavioralCloning, which similar to Nvidia's network.

Training process:

![validation loss][7.png]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

````
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 30, 30, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 13, 13, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 5, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 3, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 1, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 80)            5200        flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 80)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 40)            3240        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 40)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            656         dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 16)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            170         dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 140,625
Trainable params: 140,625
Non-trainable params: 0
````
