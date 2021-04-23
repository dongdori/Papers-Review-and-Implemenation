# FaceNet - A Unified Embedding for Face Recognition and Clustering
## 1. Problem Statement
Face Recognition is the task that decide whether 2 images are of same person or not, given image pair.
Convolutional neural network trained on classification task (e.g. classifying many people's faces) can extract embedding from image and thus execute face recognition, but its performance is poor and cross entropy loss function for classification is irrelavent to ultimate goal of face recognition, which is learning similarity and difference between faces. To solve this problem, Authors suggests novel approach called Triplet Loss, which allows model to learn degree of similarity effectively.

## 2. Approach
