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
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

In order to gauge how well the model was working, I split my image and steering angle data into a training and 10% validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set and the mse did not drop significantly after each epochs. This implied that the model was overfitting. 

But I found the data was imbalanced and the car was driving in a straight line.

![imbalanced data][1.png]

I tried to use images from left and right camera with a correction value added to steering values.


![imbalanced data][2.png]

I limited number of data between 0.0 and 0.1 to 500. 

![imbalanced data][3.png]

Lastly, If the steering value is larger than 0.15, I made steering value larger by using left or right camera. the steering value is larger than 0.2, I added the modified data twice to further combat the imbalance problem. 

![imbalanced data][4.png]


To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
