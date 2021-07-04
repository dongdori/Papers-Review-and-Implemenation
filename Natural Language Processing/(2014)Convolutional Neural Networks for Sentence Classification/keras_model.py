import numpy as np
import pandas as pd
import sys
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras import backend as K

# loading and preprocessing text data which will be used for fine-tuning


# loading pretrained GloVe word representation vectors
embeddings_index = {}
f = open('/content/drive/MyDrive/Colab Notebooks/Data science&AI/papers and Implementations/NLP/Convolutional neural networks for sentence classification/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# setting embedding matrix with words present in text data
embedding_matrix = np.random.rand((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# model
input = Input(shape=(MAX_LEN,), dtype='int32')
embeded = Embedding(VOCAB_SIZE, LATENT_DIM, input_length = MAX_LEN, trainable = True, weights = [embedding_matrix])(input)

## convolving layers with different filter sizes
x_1 = Conv1D(filters = 100, kernel_size = 3, activation = 'relu')(embeded)
x_2 = Conv1D(filters = 100, kernel_size = 4, activation = 'relu')(embeded)
x_3 = Conv1D(filters = 100, kernel_size = 5, activation = 'relu')(embeded)

## max-over time pooling
x_1 = GlobalMaxPool1D()(x_1)
x_2 = GlobalMaxPool1D()(x_2)
x_3 = GlobalMaxPool1D()(x_3)

## FC layers
x = Concatenate(axis = -1)([x_1, x_2, x_3])
x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(20, activation = 'softmax')(x)

model = keras.models.Model(input, x)
