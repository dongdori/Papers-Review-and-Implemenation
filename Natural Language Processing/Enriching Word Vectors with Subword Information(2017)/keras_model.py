# Skip-gram with negative sampling for Subword Modeling

import numpy 
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import keras.backend as K

# Config
N_SUBWORD = 300000
VOCAB_SIZE = 100000
EMBED_DIM = 300

# Model
target_subword_input = Input(shape = (None,), dtype = 'int32') ## length of subword for each word is unknown
subword_embedding = Embedding(N_SUBWORD, EMBED_DIM)(target_subword_input)
context_input = Input(shape=(1, ), dtype='int32') 
context_embedding  = Embedding(VOCAB_SIZE, EMBED_DIM)(context_input)

dot = Dot(axes = 2)([subword_embedding, context_embedding])
dot_sum = Lambda(lambda x: K.sum(x, axis = 1))(dot)

output = Activation('sigmoid')(dot_sum)

model = keras.models.Model(inputs = [target_subword_input, context_input], outputs = output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
