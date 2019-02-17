# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training/validation/testing data are distributed. It is seen that the distribution of training and testing data are very close. Also none of the dataset have a distribution that is close to uniform. Some categories (e.g. 1-5, 7-13) have much more examples than others.

![training data hist][./examples/training_hist.png]
![validation data hist][./examples/validation_hist.png]
![testing data hist][./examples/testing_hist.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not follow `As a first step, I decided to convert the images to grayscale because ...` here because I think color information helps human vision to differentiate traffic signs more easily and reduces the chance of mis-recognition, espeically for pics in low resolution. For computer vision, although it may bring some redundant information and may thus lead to the hazard of overfitting, I still think some colors, e.g. red or yellow, are very strong features that eases the classification process.

As the only step, I normalized the image data because putting all values between -1 and 1, could make the training much more easier and I do not have to worry too much on some hyper-parameters like learning rate.

Well, I did not decide to generate additional data... Because the requirement of 0.93 accuracy is not that hard for LeNet on 34799 training data, so why bother... If the requirement were 0.96, probably I will flip/rotate some pics to generate more training data. A funny thing to mention is when flipping, some "turn left" will become "turn right" and I should not forget to change the y label correspondingly.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer 				| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input 				| 32x32x3 RGB image 							| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 14x14x8					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x18	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x18					|
| Fully connected		| 450, outputs 202								|
| RELU					|												|
| dropout				|												|
| Fully connected		| 202, outputs 97								|
| RELU					|												|
| dropout				|												|
| Fully connected		| 97, outputs 43								|
| Softmax				|												|
 
It is based on the LeNet example shown in last lab on mnist. I changed input from 32x32x1 to 32x32x3, and changed output from 10 to 43. In the hidden layers I slightly added more neurons, as the input and output sizes are larger.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer which is one of the most favorite optimizer for beginners. The batch size is usually a power of 2, e.g. 64, 128, or 256. I just set it 128 considering the amount of training samples is ~30k, without much tunings. For learning rate, I think 0.001 is usually a good choice as long as the inputs are normalized to the region between -1 and 1. The number of epochs may affect underfitting or overfitting. Thus I ran it multiple times and observed. With a reasonable dropout rate, the risk of overfitting is not quite high so I did not terminate the training too early. I think 30-40 epochs give me a sound and relatively stable result.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of (sorry I forgot to record this...)
* validation set accuracy of 0.959 
* test set accuracy of 0.934

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    LeNet, mainly because there is a beautiful existing example in last chapter.
* What were some problems with the initial architecture? 
    The input and output sizes need be changed, then it is working but the validation accuracy is not as high as 0.93.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    My first thinking is that I may need make the NN a little bigger as the input/output size are bigger than the MNIST example. So I added slightly more neurons for each layer. I did not change the architecture too much, though. I think the 5x5 kernel size and 2X2 max pooling is reasonable. For 32X32 resolution, it is also hard to add one more conv layer. To reduce overfitting, dropout layers are added after fully-connected layers. That should be the biggest change in the architecture.
* Which parameters were tuned? How were they adjusted and why?
    The number of epochs. Training for too short time leads to underfitting, while for too long time leads to overfitting. So I have to stop it after some proper time. After some experimenting, I felt 30-40 epochs are relatively good for my model. Also, I tried several possible `drop_prob` values for training. I found 0.8 is a reasonable one that prevents a lot overfitting while not hurting the power of my model too much during training.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    A convolution layer works well because it gathers information from neighbor pixels together to get some useful and meaningful localized features. Then another conv layer gathers higher level info among those localized features. This makes sense because in a photo, usually the pixels next to each other are strongly related; for different areas in the photo with longer distance, the pixels are not directly related, but the extracted higher level info are related. That makes conv layers working well.
    A dropout layer randomly drops a proportion of info during training, and thus prevents relying too much on some specified info which means overfitting. Without dropout layer my validation accuracy and testing accuracy has relatively large difference (e.g. 0.94 and 0.84). On the other hand, the uncertainty from dropout layer also adds more noises and makes the training unrepeatable each time. To tell the effectivness of tuning hyperparams, I have to run multiple times to get a feeling whether it is really underfitting or overfitting, since the result is different every time.

If a well known architecture was chosen:
* What architecture was chosen? 
    LeNet.
* Why did you believe it would be relevant to the traffic sign application?
    In this project the input images are in 32x32 size, which is quite low resolution. I think a relatively simple structure (for simple I mean it is not too deep like 100+ layers) such as LeNet is a good fit on it.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    By observing the validation accuracy for the 40 epochs, I see it is in a steady state between 0.94 to 0.96. So it does not seem too underfitting at least. A checking on newly downloaded data verifies that 8 of 9 pics are correctly classified (although they are coming from different statistic distribution!) Finally, the testing and validation accuracy are not far from each other, which shows no strong sign on overfitting. Also, because of the non-deterministic nature of dropping layer, I repeated the model for a few more times to make sure it is relatively stable.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][downloaded_images/e.png] ![alt text][downloaded_images/3.png] ![alt text][downloaded_images/6.png] 
![alt text][downloaded_images/d.png] ![alt text][downloaded_images/c.png] ![alt text][downloaded_images/7.png]
![alt text][downloaded_images/8.png] ![alt text][downloaded_images/a.png] ![alt text][downloaded_images/b.png]

Frankly speaking, all of them are not hard.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image									| Prediction									| 
|:-------------------------------------:|:---------------------------------------------:| 
| General caution 						| General caution								| 
| *Wild animals crossing*				| *No passing for vehicles over 3.5 metric tons*|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Turn right ahead						| Turn right ahead								|
| Road narrows on the right				| Road narrows on the right						| 
| No entry								| No entry										|
| Bumpy road							| Bumpy road									|
| Speed limit (30km/h)					| Speed limit (30km/h)							|
| Road work								| Road work										|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.9%. This compares favorably to the accuracy on the test set of 93.4%. The only mistake in these 9 examples is the one of `Wild animals crossing`, which my model predicts as `No passing for vehicles over 3.5 metric tons` by mistake. Hmmm, it might misunderstand the cows as trucks which happens to have kind of similar shapes, LOL. To improve, I wonder we may need more training data on those two categories.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section of "`Output Top 5 Softmax Probabilities For Each Image Found on the Web`" of the Ipython notebook.

For the first image, the model is very sure that this is a General caution (probability of 1.0), and the image does contain a General caution. The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| General caution								| 
| 4.64636614e-18		| Pedestrians									|
| 2.01938473e-38		| Speed limit (30km/h)							|
| 0.00000000e+00		| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (50km/h)							|


For the 2nd image, the model is sure that this is a No passing for vehicles over 3.5 metric tons (probability of 0.998), but the image does *NOT* contain a No passing for vehicles over 3.5 metric tons. It should be an image on Wild animals crossing. The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 9.98426199e-01		| No passing for vehicles over 3.5 metric tons	| 
| 1.15780416e-03		| No passing									|
| 3.37233214e-04		| Speed limit (100km/h)							|
| 4.05284300e-05		| Right-of-way at the next intersection			|
| 3.80322526e-05		| Beware of ice/snow							|

For the 3rd image, the model is very sure that this is a Right-of-way at the next intersection (probability of 1.0), and the image does contain a Right-of-way at the next intersection. The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Right-of-way at the next intersection			| 
| 1.38714059e-21		| Pedestrians									|
| 2.77026071e-27		| Double curve									|
| 6.81889400e-29		| Children crossing								|
| 1.59922196e-29		| Speed limit (30km/h)							|

For the 4th image, the model is very sure that this is a Turn right ahead (probability of 1.0), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Turn right ahead								| 
| 2.13888123e-18		| Keep left										|
| 3.99394045e-20		| Go straight or left							|
| 2.97320760e-23		| Ahead only									|
| 1.58673928e-23		| Roundabout mandatory							|

For the 5th image, the model is very sure that this is a Road narrows on the right (probability of 1.0), and the image does contain a Road narrows on the right. The top five soft max probabilities were

| Probability 			| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Road narrows on the right						| 
| 1.30972588e-14		| Road work										|
| 9.37165300e-16		| Traffic signals								|
| 6.17939175e-16		| Double curve									|
| 2.97512408e-16		| Bicycles crossing								|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

To visualize, I used conv1 and conv2 respectively, both after max-pooling. In some features in conv1 the directions of lines/edges are clearly seen. Unfortunately in conv2, it is hardly seen any human-explainable features. Partially it is because of the low resolution (only 5x5 in this layer!). To improve, I should use the conv layers before pooling.
