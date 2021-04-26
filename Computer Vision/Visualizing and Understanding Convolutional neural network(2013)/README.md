# Visualizing and Understanding Convolutional neural network (2013)

## 1. Problem statement and Objectives
Deep Convolutional Neural network shows great performance on image recognition tasks. 
Nonetheless, We do not know *how this model can recognize image* and *what happens inside the convolutional layers*.
To improve the model, We have to figure out mechanism of current model. 
Authors suggests attaching **deConvNet layers** to each layer, thereby project(reconstruct) feature maps of each layers to input pixel space. 

#### Objectives of the paper
1. Visualizing featuremaps produced by convolutional layers through deConvnet.
2. Improve current model through analyzing results form layer feature map visualization. 
