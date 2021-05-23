import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *

# CONFIG
N_HIDDEN_STATE = 1000
S_VOCAB_SIZE = 160000
T_VOCAB_SIZE = 80000
MAX_LEN = 30
EMBED_DIM = 1000

# model for training

# input and embedding
input_sc = Input(shape = (MAX_LEN,))
sc_embed = Embedding(S_VOCAB_SIZE, 1000)(input_sc)
input_tg = Input(shape = (MAX_LEN,))
tg_embed = Embedding(T_VOCAB_SIZE, 1000)(input_tg)

# encoder
x = LSTM(N_HIDDEN_STATE, return_sequences = True)(sc_embed)
x = LSTM(N_HIDDEN_STATE, return_sequences = True)(x)
x = LSTM(N_HIDDEN_STATE, return_sequences = True)(x)
v, state_h, state_c = LSTM(N_HIDDEN_STATE, return_sequences = True, return_state = True)(x)

# decoder
y = LSTM(N_HIDDEN_STATE, return_sequences=True)(tg_embed, initial_state = [state_h, state_c])
y = LSTM(N_HIDDEN_STATE, return_sequences=True)(y)
y = LSTM(N_HIDDEN_STATE, return_sequences=True)(y)
y = LSTM(N_HIDDEN_STATE, return_sequences=True)(y)

output = TimeDistributed(Dense(T_VOCAB_SIZE, activation = 'softmax'))(y)

model = keras.models.Model([input_sc, input_tg], output)
model.summary()
