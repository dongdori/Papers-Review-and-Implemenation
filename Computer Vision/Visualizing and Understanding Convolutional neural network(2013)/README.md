# Visualizing and Understanding Convolutional neural network (2013)

## 1. Problem statement and Objectives
Deep Convolutional Neural network shows great performance on image recognition tasks. 
Nonetheless, We do not know *how this model can recognize image* and *what happens inside the convolutional layers*.
To improve the model, We have to figure out mechanism of current model. 
Authors suggests attaching **deConvNet layers** to each layer, thereby project(reconstruct) feature maps of each layers to input pixel space. 

#### Objectives of the paper
1. Visualizing featuremaps produced by convolutional layers through deConvnet.
2. Improve current model through analyzing results form layer feature map visualization. 

## 2. Approach
To project featuremaps to input pixel space, Authors suggest Unpooling, Rectifying and Deconvolving featuremaps as reversed order.

1. Unpooling : Max Pooling operation is non-invertible operation. However, we can approximate inverse operation using **switch**.   
               When max pooling is executed, location of max value in the kernel is recorded in switch. 
2. Rectification : Rectify all negatives into zero
3. Deconvolution : Apply transposed convolution operation using filters that are vertically and horizontally transposed.

Given feature maps which are output of sequence of convolutional layers, We can execute inverse operations reversely, and finally reconstruct featuremaps to input pixel space.

## 3. Experimental Results

### 3.1. 
