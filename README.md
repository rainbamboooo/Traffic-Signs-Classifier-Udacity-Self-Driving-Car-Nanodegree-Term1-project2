Traffic Signs Classifier
========================

The goals / steps of this project are the following: 

 - Load the data set 
 - Explore, summarize and visualize the data set 
 - Design, train and test a model architecture 
 - Use the model to make predictions on new images 
 - Analyze the softmax probabilities of the new images 
 - Summarize the results with a written report 
 
Load the data set
-----------------

The size of training set is 34799  
The size of test set is 12630  
The size of validation set is 4410  
The shape of a traffic sign image is 32*32  
The number of unique classes/labels in the data set is 43 

Explore, summarize and visualize the data set 
---------------------------------------------

I explored the data in training set and validation set and visualized their 
distribution. 

![](https://github.com/rainbamboooo/Traffic-Signs-Classifier-Udacity-Self-Driving-Car-Nanodegree-Term1-project2/raw/master/1.png)

Preprocess the data set 
-----------------------

I did grayscale for the data set. Because I think color is not an important factor on 
recognize the traffic signs. After doing grayscale, the input decrease from 
32\*32\*3 to 32\*32\*1. In this way, I think I can get outputs that are more accurate. 

![](https://github.com/rainbamboooo/Traffic-Signs-Classifier-Udacity-Self-Driving-Car-Nanodegree-Term1-project2/raw/master/2.png)

Design, train and test a model architecture 
-------------------------------------------

My model uses a LeNet structure:

 - Input 32\*32\*1
 - Convolutional Layer (5\*5\*1 filter, stride 1, output is 28\*28\*6) 
 - Relu 
 - Maxpooling(2\*2 kernel, 2\*2 stride) 
 - Convolutional Layer (5\*5\*6 filter, stride 1, output is 14\*14\*16) 
 - Relu 
 - Maxpooling(2\*2 kernel, 2\*2 stride) 
 - Flatten(400)
 - Fully connected layer (400->120) 
 - Fully connected layer (120->80) 
 - Fully connected layer (120->43) 
 - Output(43) 

CNN is great for this project because it can fist recognize some of the basic features 
of traffic signs and then recognize features that are more complex. Because traffic 
signs' shape is relatively simple, So two convolutional layer will be enough. 
I use a pooling layer to pick the most important feature. Meanwhile, it can decrease 
the size and make training easier. 

In the training, I fine-tune some parameters like sigma so that the model can perform 
better. 

After 11 epochs, the model reaches 93% accuracy. 

Use the model to make predictions on new images 
-----------------------------------------------

I loaded the new data and preprocessed them. 

![](https://github.com/rainbamboooo/Traffic-Signs-Classifier-Udacity-Self-Driving-Car-Nanodegree-Term1-project2/raw/master/3.png)

I think these data are relatively same as my training data, although the ¡°70km/h 
limit¡± sign seems quite strange. There are some letters in the sign that might 
cause misclassification. 

Using the previous network, the output of prediction is [32  1 41 25  4 13 17].  
I use my eyes to identify that the right label of the images should be [32, 1, 4, 25, 
38, 13, 17]. That is, I got 5 correct prediction from 7 new images, and the 
accuracy is 71.43%.  

Then I evaluated my previous model on test dataset and got an accuracy of 
91.7%. 

Comparing to that, the accuracy of new images prediction is really low, I think 
here are some reasons: 

1. The third input image is not like the normal one so it is hard to predict.

2. There are few data. If there are more, I think the total accuracy will go up to 
90% 

3. Maybe there is overfitting happen on my training process. 

Analyze the softmax probabilities of the new images 
---------------------------------------------------

The top 5 possibilities predictions for these seven images are: 

![](https://github.com/rainbamboooo/Traffic-Signs-Classifier-Udacity-Self-Driving-Car-Nanodegree-Term1-project2/raw/master/4.png)

The correct prediction still doesn¡¯t appear in the top 5 possibilities. I think the reason is that the two wrong images are somehow different from my 
training images although they are in the same class. Or I overfit my model. 
