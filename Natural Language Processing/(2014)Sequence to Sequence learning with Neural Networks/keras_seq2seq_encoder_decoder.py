import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

#0. CONFIG
N_HIDDEN_STATE = 1000
S_VOCAB_SIZE = 160000
T_VOCAB_SIZE = 80000
MAX_LEN = 30
EMBED_DIM = 1000

#1. encoder
enc_input = Input(shape = (MAX_LEN,), name = 'source_input', dtype = 'int32')
sc_embed = Embedding(S_VOCAB_SIZE+1, 1000, name = 'source_embedding')(enc_input)
x = LSTM(N_HIDDEN_STATE, return_sequences = True, name = 'encoder_lstm_1')(sc_embed)
x = LSTM(N_HIDDEN_STATE, return_sequences = True, name = 'encoder_lstm_2')(x)
x = LSTM(N_HIDDEN_STATE, return_sequences = True, name = 'encoder_lstm_3')(x)
v, state_h, state_c = LSTM(N_HIDDEN_STATE, return_sequences = False, return_state = True, name = 'encoder_lstm_4')(x)
encoder = keras.models.Model(enc_input, [state_h, state_c])


#2. decoder
dec_input = Input(shape = (MAX_LEN,), name = 'target_input', dtype = 'int32')
dec_h_state_input = Input(shape = (N_HIDDEN_STATE,), name = 'decoder_h_state_input')
dec_c_state_input = Input(shape = (N_HIDDEN_STATE,), name = 'decoder_c_state_input')
tg_embed = Embedding(T_VOCAB_SIZE+1, 1000, name = 'target_embedding')(dec_input)
y = LSTM(N_HIDDEN_STATE, return_sequences=True, name = 'decoder_lstm_1')(tg_embed, initial_state = [dec_h_state_input, dec_c_state_input])
y = LSTM(N_HIDDEN_STATE, return_sequences=True, name = 'decoder_lstm_2')(y)
y = LSTM(N_HIDDEN_STATE, return_sequences=True, name = 'decoder_lstm_3')(y)
y, dec_h_state, dec_c_state  = LSTM(N_HIDDEN_STATE, return_sequences=True, return_state = True, name = 'decoder_lstm_4')(y)
y = TimeDistributed(Dense(T_VOCAB_SIZE+1, activation = 'softmax'))(y)
decoder = keras.models.Model(inputs = [dec_input, dec_h_state_input, dec_c_state_input], outputs = [y, dec_h_state, dec_c_state])


#3. encoder-decoder for training
# get layers from encoder and decoder
sc_embedding = encoder.get_layer('source_embedding')
tg_embedding = decoder.get_layer('target_embedding')
encoder_lstm_1 = encoder.get_layer('encoder_lstm_1')
encoder_lstm_2 = encoder.get_layer('encoder_lstm_2')
encoder_lstm_3 = encoder.get_layer('encoder_lstm_3')
encoder_lstm_4 = encoder.get_layer('encoder_lstm_4')
decoder_lstm_1 = decoder.get_layer('decoder_lstm_1')
decoder_lstm_2 = decoder.get_layer('decoder_lstm_2')
decoder_lstm_3 = decoder.get_layer('decoder_lstm_3')
decoder_lstm_4 = decoder.get_layer('decoder_lstm_4')
dense_output = decoder.layers[-1]

enc_input = Input(shape = (MAX_LEN,), dtype = 'int32')
enc_embed = sc_embedding(enc_input)
x = encoder_lstm_1(enc_embed)
x = encoder_lstm_2(x)
x = encoder_lstm_3(x)
_, enc_state_h, enc_state_c = encoder_lstm_4(x)

dec_input = Input(shape = (MAX_LEN,), dtype = 'int32')
dec_embed = tg_embedding(dec_input)
y = decoder_lstm_1(dec_embed, initial_state = [enc_state_h, enc_state_c])
y = decoder_lstm_2(y)
y = decoder_lstm_3(y)
y, _, _ = decoder_lstm_4(y)
output = dense_output(y)

encoder_decoder = keras.models.Model(inputs = [enc_input, dec_input], outputs = output)
encoder_decoder.compile(loss = 'categorical_crossentropy')
