# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/X_train.png "Visualization1"
[image2]: ./examples/X_valid.png "Visualization2"
[image3]: ./examples/X_test.png "Visualization3"
[image4]: ./examples/X_test.png "Before Grayscaling"
[image5]: ./examples/X_test.png "After Grayscaling"
[image6]: ./examples/fake_data.png "Fake Data"
[image7]: ./examples/25-Road-work.jpg "Traffic Sign 1"
[image8]: ./examples/4-Speed-limit-70-km-per-h.jpg "Traffic Sign 2"
[image9]: ./examples/2-Speed-limit-50-km-per-h.jpg.jpg "Traffic Sign 3"
[image10]: ./examples/17-No-entry.jpg "Traffic Sign 4"
[image11]: ./examples/33-Turn-right-ahead.jpg "Traffic Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate and visualize summary statistics of the traffic signs data set:

* The size of training set is: 34,799
* The size of the validation set is: 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is: (32, 32, 3) 
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set (Training, validation and testing respectively). It is a bar chart showing how the data is distributed amoung the 43 classes.

![Training Data Distribution][image1]
![Validation Data Distribution][image2]
![Testing Data Distribution][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is easier for the classifier to learn from grayscaled images rather than RGB images (i.e. in our case color doesn’t have a big influence in detecting the traffic sign)  

As a last step, I normalized the image data because we want the data to have a mean of zero and an equal variance. I preformed both grayscale and normalization inthe function `grayscale_normalize`.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]
![alt text][image5]


I decided to generate additional training data because from the bar chart presented earlier, it is very obvious that the data is unbalanced. So I found the mean of the images per class (809) and used it as a stopping criteria for augmenting the data to each class with data less than the average. Note that since there are classes with 180 images, the maximum number of images we can add is 180 to result in a total of 360 (doubling the data) which is still below the average but was sufficient to raise the accuracy of the classifier.

To generate a fake image from an original one, the original data was rotated by an angle $\in$ $[-25,25]$ and translated by a shift in x and y \in$ $[-5,5]$.

Here is an example of an original image and an augmented image:

![alt text][image5]
![alt text][image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					| Using TensorFlow function `tf.nn.relu()`		|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					| Using TensorFlow function `tf.nn.relu()`		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Dropout   	      	| Using `tf.nn.dropout()`, Keep_prob = 0.5  	|
| Fully connected 01	| Input = 400, Output = 120      				|
| RELU					| Using TensorFlow function `tf.nn.relu()`		|
| Dropout   	      	| Using `tf.nn.dropout()`, Keep_prob = 0.5  	|
| Fully connected 02	| Input = 120, Output = 84      				|
| RELU					| Using TensorFlow function `tf.nn.relu()`		|
| Dropout   	      	| Using `tf.nn.dropout()`, Keep_prob = 0.5  	|
| Fully connected 03	| Input = 84, Output = 43       				|
|						|												|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* Batch size: 128
* Number of epochs: 100 epochs
* Learning rate: 0.001
* Objective function: Softmax cross entropy 
* Optimizer: AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.971
* validation set accuracy of 0.956 
* test set accuracy of 0.946

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * LeNet, because it showed a good performance on grayscaled images (handwritten characters) of size 32x32.
    
* What were some problems with the initial architecture?
    * It performed poorly on the traffic sign dataset (Training accuracy of 0.89)  
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * I preformed a dropout after the second convolution layer and the first and second fully connected layers, and that is to enhance the performance of the classifier and avoid overfiting.   
    
* Which parameters were tuned? How were they adjusted and why?
    * Increasing the number of epochs from 10 to 100 helped in increasing both the training and validation accuracies.
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * As I mentioned previously, having a dropout layer is a form of regularization, it prevents overfitting and improves the performance. A convolution layer would be very important in our case because it workes as a feature extractor and different traffic signs definitely have different set of features that distinguish them apart.       

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Speed limit (70km/h)  | Speed limit (20km/h) 							|
| Speed limit (50km/h)	| Speed limit (20km/h)							|
| No entry	      		| No entry  					 				|
| Turn right ahead		| Turn right ahead     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares less favorably to the accuracy on the test set of 94.6 %

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the end of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work (probability of 0.84), and the image does contain a stop sign. The top five softmax probabilities (rounded, don’t some up to one) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84         			| Road work   									| 
| .007     				| Dangerous curve to the right 					|
| .003					| Traffic signals   							|
| .002	      			| Bicycles crossing  							|
| .002				    | Bumpy road             						|

For the second image, the model is very sure that this is a speed limit (20km/h) (probability of 0.99), and the image doesn’t  contain a speed limit (20km/h) but a speed limit (70km/h). The top five softmax probabilities (rounded, don’t some up to one) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| speed limit (20km/h)   						| 
| .01     				| Speed limit (30km/h)      					|
| .00					| Speed limit (70km/h)   						|
| .00	      			| Keep left         							|
| .00				    | Dangerous curve to the left  					|

For the third image, the model is not quite sure that this is a speed limit (20km/h) (probability of 0.49), and the image doesn’t  contain a speed limit (20km/h) but a speed limit (50km/h). The top five softmax probabilities (rounded, don’t some up to one) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| speed limit (20km/h)   						| 
| .25     				| Speed limit (30km/h)      					|
| .08					| Roundabout mandatory   						|
| .04	      			| Vehicles over 3.5 metric tons prohibited      |
| .04				    | End of speed limit (80km/h)  					|

For the fourth image, the model is 100% sure that this is a No entry (probability of 1.0), and the image does contain a no entry sign. The top five softmax probabilities (rounded, don’t some up to one) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry              						| 
| .00    				| Vehicles over 3.5 metric tons prohibited      |
| .00					| Roundabout mandatory   						|
| .00	      			| Go Straight or Left   						|
| .00				    | Stop                      					|


For the fifth image, the model is not quite sure that this is a Turn right ahead sign (probability of 0.41), and the image does contain a right ahead sign. The top five softmax probabilities (rounded, don’t some up to one) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .41         			| Turn right ahead        						| 
| .23    				| End of speed limit (80km/h)                   |
| .11					| Keep left             						|
| .06	      			| Stop                   						|
| .06				    | Roundabout mandatory        					|


I really didn’t expect the classifier to misclassify the speed limit (70km/h) and speed limit (50km/h) signs because there classes (2 and 4 respectively) contain one of the highest training images. It might be due to light conditions of the images, the speed limit (70km/h) sign has a very high contrast and the speed limit (50km/h) sign has a low contrast.  

