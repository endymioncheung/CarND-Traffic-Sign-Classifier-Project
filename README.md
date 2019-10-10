
# **Traffic Sign Recognition** 

## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

Overview
---
In this project, the deep neural networks and convolutional neural networks are used to classify traffic signs. The convolutional neural networks model are trained with validation datasets so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, some random test images of German traffic signs found online will be used to predict and classify the traffic sign.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Dataset

[Pickled datasets for the German traffic signs](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). The images in this pickled dataset is resized to 32x32. It contains a training, validation and test set.

[//]: # (Image References)
[training_dataset_histo]: ./report_img/histo_training_dataset.png "Training Dataset Visualization"
[validation_dataset_histo]: ./report_img/histo_validation_dataset.png "Validation Dataset Visualization"
[test_dataset_histo]: ./report_img/histo_testing_dataset.png "Testing Dataset Visualization"

[trimmed_training_dataset_histo]: ./report_img/histo_trimmed_training_dataset.png "Trimmed Training Dataset Visualization"
[trimmed_validation_dataset_histo]: ./report_img/histo_trimmed_validation_dataset.png "Trimmed Validation Dataset Visualization"

[high_res_Speed_limit_80_km_hr]: ./test_images/original_high_res/Speed_limit_80_km_hr.jpg "Class #5 Speed Limit 80km/hr"
[high_res_Speed_limit_100_km_hr]: ./test_images/original_high_res/Speed_limit_100_km_hr.jpeg "Class #7 Speed Limit 100km/hr"
[high_res_Yield_sign]: ./test_images/original_high_res/Yield_1.jpg "Class #13 Yield"
[high_res_Stop_sign]: ./test_images/original_high_res/Stop_sign_1.jpg "Class #14 Stop sign"
[high_res_No_entry]: ./test_images/original_high_res/No_Vehicles.png "Class #17 No Entry"
[high_res_Bumpy_road]: ./test_images/original_high_res/Bumpy_road_sign.jpg "Class #22 Bumpy road sign"

[class_5_Speed_limit_80_km_hr]: ./test_images/class_5_Speed_limit_80_km_hr.jpg "Class #5 Speed Limit 80km/hr"
[class_7_Speed_limit_100_km_hr]: ./test_images/class_7_Speed_limit_100_km_hr.jpg "Class #7 Speed Limit 100km/hr"
[class_13_Yield_1]: ./test_images/class_13_Yield_1.jpg "Class #13 Yield"
[class_14_Stop_sign_1]: ./test_images/class_14_Stop_sign_1.jpg "Class #14 Stop sign"
[class_17_No_Entry]: ./test_images/class_17_No_Entry.jpg "Class #17 No Entry"
[class_22_Bumpy_road_sign]: ./test_images/class_22_Bumpy_road_sign.jpg "Class #22 Bumpy road sign"
[trimmed_train_trimmed_valid_classification]: ./report_img/trimmed_train_trimmed_valid_classification.png "Traffic sign classification based on the trimmed training and trimmed validation datasets"

[10_random_traffic_signs]: ./report_img/10_random_traffic_signs.png "10 random traffic signs"
[10_random_traffic_signs_grayscale]: ./report_img/10_random_traffic_signs_grayscale.png "10 random traffic signs (grayscale)"
[new_test_images]: ./report_img/new_test_images.png "6 new traffic signs test images found on the web"



[augmentation_comparison_roadwork_sign]: ./report_img/augmentation_comparison_roadwork_sign.png "Augmentation Comparison (Roadwork)"
[augmentation_comparison_120_km_hr]: ./report_img/augmentation_comparison_120_km_hr.png "Augmentation Comparison (120 km/hr)"


---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

```python
# Training, validation and testing datasets
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test,  y_test  = test['features'], test['labels']

# Number of training examples
n_train = len(X_train)
# Number of validation examples
n_valid = len(X_valid)
# Number of testing examples.
n_test = len(X_test)
# Shape of an traffic sign image?
image_shape = X_train[0].shape
# Number of unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3) in other words 32x32px in color channels
* The number of unique classes/labels in the data set is 32

#### 2.Exploratory visualization of the datasets.

Here is an exploratory visualization of the original data set. It is a bar chart showing how the data distributes across the 43 traffic sign classes for the original dataset.

![][training_dataset_histo]
![][validation_dataset_histo]
![][test_dataset_histo]

As shown above, the original training dataset has a relative uneven distribution amongst the classification classes. In order for the neural networks to have a more accurate, reliable prediction in image classification, ideally we should aim for a large dataset of even distribution dataset so that the network can learn each image. In additional, it is also as  important to have a similar data distribution in the validation dataset like the training dataset.

Assuming both training and validation datasets have similar distribution, which is the case for the given German traffic sign dataset, an easy and effective way to acheive even distribution is by clipping both training and validation dataset to a lower maximum samples per class, this is the easiest pre-process technique to "balance" the dataset distribution. As shown below, now the training and validation dataset look more even now.

![][trimmed_training_dataset_histo]
![][trimmed_validation_dataset_histo]

Instead of using the given dataset for the training and validation, I used the more uniform distributed "trimmed" trainng (limited to 1250 samples per class) and validation (limited to 150 samples per class) datasets shown above for a slightly faster and robust training.

### Design and Test a Model Architecture

**Preprocess data #1: Convert images to Grayscale**

As a first step, I decided to convert the images to grayscale because it helps the network to train faster/easier to learn about the traffic signs as the color is not a factor in determining the traffic signs.

Here is an example of a traffic sign image before and after grayscaling.

![][10_random_traffic_signs]
![][10_random_traffic_signs_grayscale]

**Preprocess data #2: Noramlize images**
*Note the order of the grayscaling or normalizing does not matter*

As a last step, I normalized the image data because it helps to ensure that each image in the dataset to have a similar data distribution. This makes convergence much faster while training the network. 

**Preprocess data #3: Generate Augmented images** (Dataset created but not used for training the network for image classification

To generate and add more data to the original data set (for better training generalization), one could use the following data augementation techniques (horizontal flips, vertical flips, flips in both x and y directions, random translation, random rotation, random brightness) because it adds noises/jitter to the training dataset while retaining the invariant features of the dataset

Here is an example of an original image and an augmented image for two randomly selected traffic signs:

![][augmentation_comparison_120_km_hr]
![][augmentation_comparison_roadwork_sign]

Two variants of the augemented datasets were created to experiment the benefits of generalizing the network training:
1. Original dataset + generated augmented images (horizontal flips, vertical flips, flips in both x and y directions, random translation, random brightness) based on the original dataset

2. Trimmed dataset (limit up to 1250 images per class) + generated augmented images(horizontal flips, vertical flips, flips in both x and y directions, random translation, random brightness) based on the trimmed dataset

Attempts were made to improve the validation accuracy by adding the augmentated images to the original and the trimmed training and validation datasets, however the validation accuracy actually it is about 10% lower. More investigation is required to understand this issue. Perhaps more layers (more training parameters) are needed for the convolution neural networks so that more useful features can be identified and distinguish between different road signs.


#### 2. Final model architecture

My model is based off the original LeNet convolution neural networks from the lab example with minor modifications:
- changed the input depth of 1 instead 3 to accept grayscale image rather than RGB image
- changed the output class of 43 instead of 10 to classify 43 unique traffic signs rather than 10 unique digits
- added dropout layers immediately after the RELU activation function for better training generalization

My final model is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| valid padding, 1x1 stride, outputs 28x28x6 	|
| RELU					| Lighweight activation function				|
| Max pooling	      	| input 28x28x6, 2x2 stride, outputs 14x14x6   	|
| Convolution 5x5     	| valid padding, 1x1 stride, outputs 10x10x16 	|
| RELU					| Lighweight activation function				|
| Max pooling	      	| input 10x10x16, 2x2 stride, outputs 5x5x16   	|
| Flattern  	      	| input 5x5x16 outputs 400    			       	|
| Fully connected		| input 400 outputs 120							|
| RELU					| Lighweight activation function				|
| **Dropout**			| **Dropout = 0.5**                				|
| Fully connected		| input 120 outputs 84							|
| RELU					| Lighweight activation function				|
| **Dropout**			| **Dropout = 0.5**                				|
| Fully connected		| input 84 outputs 43							|


#### 3. How you trained your model? The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Optimizer = AdamOptimizer**

To train the model/network weights, I chose AdamOptimizer rather than the traditional SGD (stochastic gradient descent) to update the network weights for its ability to decay the learning rate accordingly, also known as adaptive moment estimation. The learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

**Learning rate = 0.001**

Learning rate is selected to be 0.001. In general this chosen learning rate yields stable validation accuracy convergence without excessive long training time. A smallear learning rate in this application (i.e. 0.0005) does not show any significant improvement in the final validation accuracy. This makes sense for the AdamOptimizer adaptive optimizer chosen above, which has algorithm to adaptively decay the learning rate if needed.

If a SGD optimizer is chosen, it is expected the learning rate has a greater impact on the training loss and validation accruacy.

**Mini-batch size = 256**

*"Small batch size = fast converge to validation accuracy but larger variance of accuracy"*

Given the current dataset size, the mini-batch size of 256 was chosen for training stability and generalization while still ensuring the validation accuracy to converge. Despite a smaller batch size such as 128 or 64 can be chosen to achieve the minimum of 93% validation accuracy, higher initial validation accuracy and converges faster, however at the expense of overfitting data to the given training and validation set and hence not general enough to be used with new traffic sign images that the clasifier has never seen before.

**Number of epochs ~= 25**

The number of epochs were chosen "roughly" to be 25 to allow network weights to be optimized for the target 93% validation accuracy and then later was fine-tuned with a few more epochs up to 5. This technique is known as **early stopping** which helps the neural networks to prevent overfiting training data for unnecessary training bias such as uneven distritbution in the training dataset. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.959 
* test set accuracy of 0.915

The iterative approach that I've taken to achieve traffic signs classification was to choose [LeNet network] (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) for training and validating the image datasets, which is a well known convolution neutral network architecture that demonstrates high accuracy of classification of handwritten numbers and other image classification applications. Given the similar of image classification application, the LeNet can be re-used by simply modification to recognize up to 43 different classes of images, rather than just 10 different numbers. To enhance the robustness of traffic sign classification with the original LeNet (i.e. reduce tendency of overfitting when the network bias towards unnecessary features), **two drop out operations** are applied immediately to the (RELU) activation function to introduce noises to reduce the bias during the neural networks (weight) training.

Apart to achieve the minimum of 93% or greater validation accuracy, I also aim to have a robust training that works with the real German traffic sign images that the classifier had never seen before. For example, I repeated train the networks from scratch within a give number of epochs, in this case ~25 a number of times to ensure the classification accuracy is consistent.

**Strategy for tuning the hyper-parameters learning rate, mini-batch size, number of epochs**
1. Start off the network training a set of default values with the given German traffic signs  training and validation datasets:
   - learning rate = 0.001 (can be in the range of [0..1])
   - mini-batch size = 128
   - number of epochs = 10
2. Initially experiment with each of hyperparameter with different values for quick convergence to reach valdation accuracy of 93%. 
3. Created a visualization function and observe the training loss and validation accuracy as the number of epochs increases. Observe the characteristics, trends of the both training losses and validation accuracy. With the hyperparameter value changes:
    - How does it affect the initial validation accuracy?
    - How does it affect the validation accuracy convergence?
    - What is the maximum validation accuracy?
    - How stable is the validation accuracy with the epoch increase?
4. Identify the best values for hyperparameters and repeat it for few more times to ensure that similar validation accuracy are repeatable
5. For the number of epochs tuning, run a small number of epochs in step of say 10/20 epoch, observe if the validation accuracy is still increasing. If this is the case, it means the network is still underfitting, in other words it needs more training, so I save the training session then progressively train with a small number of 10/5/1 from the previously saved session until the goal validation accuracy slightly exceed 93%. Else, go back to the second last trained session and try with an even smaller epcoh 
6. Now that the goal validation accuracy is reached, observe how well the prediction of the traffic sign classifier with the new traffic sig test images found on the web.
    - How is the test accuracy on these images?
    - How is the accuracy of the first 5 guesses of each new test image? If they are 
7. Experiment the model training with different combinations of (training and validation) datasets variants
    - Original datasets
    - Trimmed datasets limited up to 1250 sample images per traffic sign class
    - Trimmed datasets + augmented images (flips, random translation, random zoom, random brigtness)
    - "Balanced" datasets that include the augmented images to the underfitting class of the trimmed dataset. All traffic sign classes are 1250 counts.

If there is a large deviation from the validation accuracy to the test accuracy, I'll consider these common techniqes to prevent overfitting then starting all over again with the step 1 of the training strategy:
- Regularization: dropout is added to each post activation function of the original LetNet
- Early stopping: observe when the validation accuracy no longer improves
- Augmentation: helps to build a larger dataset by generating multiple variants of the original data such as flipping the image along horizontal-axis, vertical-axis, translation, rotation, brightness adjustment.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][high_res_Yield_sign]
![alt text][high_res_Speed_limit_80_km_hr]
![alt text][high_res_Speed_limit_100_km_hr] 
![alt text][high_res_No_entry]
![alt text][high_res_Stop_sign]
![alt text][high_res_Bumpy_road]

These images are manually cropped and resized to 32x32 pixels, like one shown below.

![][new_test_images]

Then these 6 test images go through the same grayscaling and normalization pre-processing applied earlier with the other datasets, so the image dimensions reduces from 32x32x3 to 32x32x1 as the modified LeNet takes grayscale image.

For the rationale selecting the test images for the German traffic signs, I picked three traffic sign images that are higher and another three lower than average count of traffic signs per class. The sequence of displaying the six images are sorted in descending order (starting with the highest image count per traffic sign class first to show the easiest to more challenging image classifications).

Given the high image counts of the traffic sign images (yield, speed limit 80 km/hr, speed limit 100 km/hr) in the training dataset are expected to be easier for the traffic sign classifier to correctly classify.

Conversely, the later three traffic sign images (no entry, stop sign, bumpy road sign) are likely to have difficulties in correctly predicting the images as it has a lower image counts in the training dataset.

With the two speed limit signs (80km/hr and 100km/hr) in the test images, interestingly it was found that sometimes the trained models even with the same number of epochs (i.e. 25) is not able to always classify both speed signs correctly at the same time. Typically it always correctly predicts either the 80km/hr or 100km/hr but not always both. Give the validation accuracy and validation of the trained model is reasonablely good, which is 0.959 and 0.915 respectively, it is possible that the trained model could still be overfitted. Since I had already implemented most of the common prevent overfitting techniques such as implementing dropout, shuffled the dataset, pre-processed the dataset to make it the data more uniform distributed, increased batch size and early stop. In future, I would like to experiment it with adding a few more convolution layer to see if it can help the model to distinguish between the different speed limits signs easier.

It was expected that the last image - bumpy road signs fails to predict correctly by the image classifier most of the time because this traffic sign class has one of the lowest counts of samples per class and it is almost half of the average sample counts per class. In other words, it didn't have much exposure to the model to learn predicting this traffic sign. One potential solution to improve the prediction of the traffic sign with low sample counts per class is to generate augment image data and add it to this specific class to increase the exposure of this traffic sign to be learnt.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        | Image counts |     Prediction	        					| 
|:---------------------:|:------------:|:------------------------------------------:| 
| Yield          		| 1250         | Yield				    					| 
| Speed Limit (80km/hr) | 1250         | Speed Limit (80km/hr) 	    				|
| Speed Limit (100km/hr)| 1250         | Speed Limit (100km/hr)		    			|
| No Entry	      		| 990          | No Entry   			 		    		|
| Stop Sign 			| 690          | Stop Sign     						    	|
| Bumpy Road Sign 		| 330          | Bicycle crossing   					    |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. Despite the test accuracy with the new test images is lower than the accuracy on the test set of 91.5%, nonetheless it is still an accurate traffic sign classifier in general because predicting the bumpy road sign is a challenging task without additional augmented data.

The accuracy on predicting on the test set can be subjective sometimes because one might pick the traffic signs with the highest count of images in the training dataset, which could give the false impression that it works almost 100% accuracy. Whereas on the other extreme, one can deliberately make the classifier performance looks really bad by  picking all of the most challenge traffic signs to recognize, for example the bumpy roads that have limited number of training images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is as follows:

```python
# Define Softmax probabilities for multi-class classifications operation
softmax_logits = tf.nn.softmax(logits)

# Define top 5 Softmax probabilities operation
top_k = tf.nn.top_k(softmax_logits, k=5)  

saver = tf.train.Saver()
with tf.Session() as sess:
    # Restore the previous training session
    saver.restore(sess,saved_model)
    
    # Test Accuracy
    test_accuracy = evaluate(X_new_test, new_test['labels'])
    print('')
    print("Test Accuracy for the new images= {:.3f}".format(test_accuracy))
    print('')
    
    # Run the operations to predict the closest match traffic sign classifications
    softmax_results = sess.run(softmax_logits,feed_dict={x: X_new_test, keep_prob: 1.0})
    top_k_results = sess.run(top_k, feed_dict={x: X_new_test, keep_prob: 1.0})

# Softmax probabilities
top_k_softmax = top_k_results[0]

# Array index of the corresponding Softmax probabilities
top_k_indices = top_k_results[1]
```

![][trimmed_train_trimmed_valid_classification]

For the first image, the model is relatively sure that this is a stop sign (probability of 100%), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%         			| Yield sign   									| 
| 0.0%    				| No vehicle        							|
| 0.0%					| Ahead only             						|
| 0.0%	      			| Priority road    				 				|
| 0.0%				    | Go straight or right 							|


For the second image, the model is relatively sure that this is a speed limit 80km/hr sign (probability of 99.1%), and the image does contain a speed limit 80 km/hr sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.1%        			| Speed limit (80km/hr) 						| 
| 0.9%    				| Speed limit (60km/hr)     					|
| 0.0%					| End of speed limit (80km/hr)  				|
| 0.0%	      			| Speed limit (50km/hr)     	 				|
| 0.0%				    | Speed limit (30km/hr)     	 				|

This time it is slightly less certain than the first image because there are other similar speed limit signs with the same number of counts, most of them were capped at 1250 samples, which means that the model may have difficulty extracing useful feature to distinguish the subtle difference between different speed limits.

For the third image, the model is relatively sure that this is a speed limit 100km/hr sign (probability of 77.1%), and the image does contain a speed limit 100 km/hr sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 77.1%        			| Speed limit (100km/hr) 						| 
| 24.9%    				| Speed limit (50km/hr)     					|
| 1.9%					| Speed limit (30km/hr)  		        		|
| 1.8%	      			| Speed limit (80/hr)       	 				|
| 0.0%				    | Speed limit (120km/hr)                        | 	 				

Similar prediction results were also observed for the fourth and the fith image that the first guess is correct and with a high certainty (i.e. greater than 90%)

For the last image - bumpy road sign, the classifier fail to predict within the first 5 guesses. The classification results and guesses are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 79.3%        			| Bicycle crossing      						| 
| 24.9%    				| Left turm ahead           					|
| 1.9%					| Children crossing  	    	        		|
| 1.8%	      			| Road narrows on right       	 				|
| 0.0%				    | Slippery road                                 | 	 				

This indicates that the bumpy road signs class is underfit. Adding augmented data to this class in the training set may help the model to learn better with traffic signs like this with one of the lowest count of images in the training dataset.
