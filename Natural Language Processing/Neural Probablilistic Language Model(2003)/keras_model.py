import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *


def model(vocab_size, hidden_size, m, n): # m:dimension of distributed representation, n: window size
  tokens = Input(shape = (n,))
  embeddings = Embedding(vocab_size, m)(tokens)
  #concatenating all input embeddings
  x = Reshape(target_shape = (m*n, 1))(embedding)  
  hidden = Dense(h, activation = 'tanh')(x)
  o1 = Dense(vocab_size, activation = 'linear')(hidden)
  o2 = Dense(vocab_size, activation = 'linear')(x)
  o = Add()([o1, o2])
  #softmax activation function to output conditional probability distribution
  o = Activation('softmax')(o)
  model = keras.models.Model(inputs = input, outputs = o)
  model.summary()
  return model
    
  
