# Convolutional Neural Networks for Sentence Classification
## 1. Background
Words can be represented as lower dimensional vector via learning representations thorugh language models such as *Skip-gram, CBOW and GloVe*.
Those pre-trained word vectors shows state-of-art performance on many kinds of sentence classification benchmarks. 
Author suggests novel approach for sentence classification based on 1D convolution and pre-trained word vectors(Word2Vec).

## 2. Approach

### 2.1. Sentence Representation Through Embedding layer
- Let each words to be represented as **r-dimensional vector**.
- Let each sentences are tokenized and padded as length of **MAX_LEN**.

Then each sentences can be represented as **MAX_LEN x r** dimensional matrix.

### 2.2. 1D Convolution layer
- Create 100 feature maps using 1D-convolution filter size of 3, 4, 5 respectively
- Max-overtime pooling : pooling only maximum values of each 300 feature maps

### 2.3. Regularization and Fully connected layer
- Apply Dropout layer with p = 0.5
- Apply L2-regularization
