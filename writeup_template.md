# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes. The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting . The data was pre-processed to generate more data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .
Udacity sample data was used for training.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was to use the Nvidia architecture since it has been proven to be very successful in self-driving car tasks. The architecture was recommended in the lessons and it's adapted for this use case.
In order to test how well the model was working, I split my image and steering angle data into a training and validation set. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps. I uses 20% as split criteria for data.


#### 2. Final Model Architecture

The final model architecture code is below :

```
        model = Sequential()
        model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
        model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
```


Here is a visualization of the architecture:
![alt text](nn.png)


#### 3. Creation of the Training Set & Training Process

To create the training data, I used the Udacity sample data as a base. For each image, normalization would be applied before the image was fed into the network. In my case, a training sample consisted of four images:

Center camera image
Center camera image flipped horizontally
Left camera image
Right camera image

I found that this was sufficient to meet the project requirements for the first track.
