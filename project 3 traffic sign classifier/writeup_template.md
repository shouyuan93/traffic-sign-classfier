# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/original_train_set.png "Visualization"
[image2]: ./writeup_img/extended_train_set.png 'visual'
[image3]: ./test_images/12_test.jpg
[image4]: ./test_images/13_test.jpg
[image5]: ./test_images/14_test.jpg
[image6]: ./test_images/35_test.jpg
[image7]: ./test_images/4_test.jpg
[image8]: ./examples/placeholder.png "Traffic Sign 5"



### Data Set Summary & Exploration

#### 1. the original data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,32)
* The number of unique classes/labels in the data set is 43

#### 2. extend more data to improve the robust of classifier

Here is an exploratory visualization of the data set. The Image below is a bar chart showing how the data distribute

[image1]: ./writeup_img/original_train_set.png "Visualization"
then I use the random rotation and tranlation to generate fake data, in order to better the train set distribution.
[image2]: ./writeup_img/extended_train_set.png'visual'


#### 3. do normalize to all the input image

I use 3 channel Image (32,32,3),as imput, and create 35 channel output, because the first layer character is very importent for the later classifier. The train set, validation and test set all need to do normalize, rescale the rgb value into (0,1)

### Design and Test a Model Architecture


#### 1.  Final model architecture

My final model is basised on LeNet, consisted of the following layers:

CNN,
Input:32x32x3 RGB image
##### First layer
First layer, convolution: 3x3 filter, 1x1 stride, VALID padding, input 32x32x3, output 30x30x35
activate funktion:Relu
#####  max pooling
Max pooling, 2x2 filter, 2x2 stride, VALID padding, input 30x30x35, output 15x15x35
##### Second layer
Second layer, convolution: 3x3 filter, 1x1 stride, VALID padding, input 15x15x35, output 13x13x45
activate funktion:Relu
##### max pooling
Max pooling, 2x2 filter, 2x2 stride, VALID padding, input 13x13x24, output 6x6x45
##### Third layer
Third layer, convolution: 3x3 filter, 1x1 stride, VALID padding, input 6x6x45, output 4x4x45
activate funktion:Relu
##### Flatten
flatten convert the img 4x4x55 to 880
##### Fourth layer
Fourth layer, fully connected, input 880, output 120
activate funktion:Relu
##### Fifth layer
Fifth layer, fully connected, input 120, output 84
activate funktion:Relu
##### Sixth layer
Sixth layer, fully connected, input 84, output 43



#### 2. Model training. 

I use cross entropy to describe the loss funktion, and use AdamOptimizer to decrease the loss funktion
learn rate: 0.001
epoch: 25
batch_size: 128


#### 3. Result. 

My final model results were:
* validation set accuracy 
* test set accuracy
EPOCH 1 ...
Test Accuracy = 0.925
Validation Accuracy = 0.856

EPOCH 2 ...
Test Accuracy = 0.966
Validation Accuracy = 0.889

EPOCH 3 ...
Test Accuracy = 0.977
Validation Accuracy = 0.908

EPOCH 4 ...
Test Accuracy = 0.983
Validation Accuracy = 0.911

EPOCH 5 ...
Test Accuracy = 0.978
Validation Accuracy = 0.909

EPOCH 6 ...
Test Accuracy = 0.988
Validation Accuracy = 0.922

EPOCH 7 ...
Test Accuracy = 0.990
Validation Accuracy = 0.944

EPOCH 8 ...
Test Accuracy = 0.993
Validation Accuracy = 0.936

EPOCH 9 ...
Test Accuracy = 0.992
Validation Accuracy = 0.932

EPOCH 10 ...
Test Accuracy = 0.994
Validation Accuracy = 0.937

EPOCH 11 ...
Test Accuracy = 0.994
Validation Accuracy = 0.942

EPOCH 12 ...
Test Accuracy = 0.997
Validation Accuracy = 0.949

EPOCH 13 ...
Test Accuracy = 0.996
Validation Accuracy = 0.945

EPOCH 14 ...
Test Accuracy = 0.997
Validation Accuracy = 0.941

EPOCH 15 ...
Test Accuracy = 0.995
Validation Accuracy = 0.946

EPOCH 16 ...
Test Accuracy = 0.998
Validation Accuracy = 0.958

EPOCH 17 ...
Test Accuracy = 0.995
Validation Accuracy = 0.951

EPOCH 18 ...
Test Accuracy = 0.998
Validation Accuracy = 0.956

EPOCH 19 ...
Test Accuracy = 0.996
Validation Accuracy = 0.956

EPOCH 20 ...
Test Accuracy = 0.995
Validation Accuracy = 0.943

EPOCH 21 ...
Test Accuracy = 0.997
Validation Accuracy = 0.960

EPOCH 22 ...
Test Accuracy = 0.997
Validation Accuracy = 0.953

EPOCH 23 ...
Test Accuracy = 0.996
Validation Accuracy = 0.951

EPOCH 24 ...
Test Accuracy = 0.998
Validation Accuracy = 0.962

EPOCH 25 ...
Test Accuracy = 0.999
Validation Accuracy = 0.960


an iterative approach,
* What was the first architecture that was tried and why was it chosen?
my first model just have 5 layer, and 

firstly, I just use grayscale imp input, and 20 channel output in the first layer, but it cannot achieve the accuracy, then I use the rgb 3 channel img, it contain more information.  And add output channel, do the same for the 2nd, 3nd layer (add width). 
I think the reason is lack of information on the bottom, so its necessery to add width

secondly, I add one more fully conected layer, because the top character is almost relevate with location, so one more fully conected layer is necessery
 
the final architechture() works good, and according to the Validation Accuracy record, there is no overfitting, because the accuracy is continued shock rise.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image3]: ./test_images/12_test.jpg 'priority road'
[image4]: ./test_images/13_test.jpg 'Yield'
[image5]: ./test_images/14_test.jpg 'Stop'
[image6]: ./test_images/35_test.jpg 'ahead only'
[image7]: ./test_images/4_test.jpg 'speed limit 70km/h'

All the test_img change a littel bit perspective from envisage
The 7nd image is false classfied, but I think this because， there is only one sample, so it doesn't make meaning

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| priority road    		| priority road 								| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| Ahead only    		| Ahead only					 				|
| 70km/h limit			| Bumpy road         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Because the samples are too small, so in statistics doesn't make mean

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first 4 images, the model is completely sure(100%), and the last image is false 
first priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| priority road     							| 
| 0     				| 29    										|
| 0 					| 5 											|
| 0 	      			| 14        					 				|
| 0 				    | 31                							|

second Yield

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Yield             							| 
| 0     				| 1     										|
| 0 					| 14 											|
| 0 	      			| 2         					 				|
| 0 				    | 6                 							|

third Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Stop              							| 
| 0     				| 1     										|
| 0 					| 25 											|
| 0 	      			| 28        					 				|
| 0 				    | 18                							|

fourth Ahead only

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Ahead only           							| 
| 0     				| 37     										|
| 0 					| 36 											|
| 0 	      			| 34        					 				|
| 0 				    | 33                							|


fifth 70km/h 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100        			| Bumpy road           							| 
| 0     				| 9     										|
| 0 					| 41 											|
| 0 	      			| 39        					 				|
| 0 				    | 37                							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


