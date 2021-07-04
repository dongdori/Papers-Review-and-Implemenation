# FaceNet - A Unified Embedding for Face Recognition and Clustering
## 1. Problem Statement
Face Recognition is the task that decide whether 2 images are of same person or not, given image pair.
Convolutional neural network trained on classification task (e.g. classifying many people's faces) can extract embedding from image and thus execute face recognition, but its performance is poor and cross entropy loss function for classification is irrelavent to ultimate goal of face recognition, which is learning similarity and difference between faces. classification model can only perform with closed set of face data.

To solve this problem, Authors suggests novel approach called Triplet Loss, which allows model to learn degree of similarity effectively.

## 2. Approach

### 2.1. Metric Learning - Triplet loss

<img width="351" alt="triplet loss" src="https://user-images.githubusercontent.com/70640345/113648680-f6e82780-96c7-11eb-84ab-51b67b03d4b3.png">

Above is the formula of Triplet loss, where E<sub>A</sub>, E<sub>P</sub>, E<sub>N</sub> denotes embeddings of Anchor image, Positive image and Negative image respectively.

Triplet loss induces model to learn that L2 distance between E<sub>A</sub> and E<sub>N</sub> have to be larger than distance between E<sub>A</sub> and E<sub>P</sub> at least as big as <b>&alpha;</b>, which is hyperparameter.

During training, encoder learns degrees of similarity between images and finally is able to return sensible embedding of certain image.
I fine-tuned 2 image models with additional linear layer on top of the pretrained layer.
