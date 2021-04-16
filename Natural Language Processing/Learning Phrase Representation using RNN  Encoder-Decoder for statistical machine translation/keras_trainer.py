import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import *

def RNN_encoder_decoder_trainer(K,
                                max_len,
                                n_hidden_units = 1000,
                                dim_embed = 100):
    '''
    K : size of Vocabulary
    max_len : max length of input tokens(rests are padded with 0)
    n_hidden_units : # hidden units in GRU
    dim_embed : Dimension of Embedding vectors of words
    
    '''
    ############################ Encoder #######################################
    # input sentence tokens are labeled to int
    encoder_input = Input(shape = (max_len,), name = 'encoder_input_layer')
    enc_emb = Embedding(input_dim = K, output_dim = dim_embed, input_length = max_len,
                      name = 'input_Embeddings')(encoder_input)
    m_x = GlobalAveragePooling1D(name = 'm_x')(enc_emb)
    h_enc = GRU(n_hidden_units, activation = 'tanh')(enc_emb)
    ############################ Decoder #######################################
    # input sentence tokens are labeled to int(decoder inputs are lagged)
    decoder_input = Input(shape = (max_len,), name = 'decoder_input_layer')
    dec_emb = Embedding(input_dim = K, output_dim = dim_embed, input_length = max_len,
                      name = 'output_Embeddings')(decoder_input)
    decoder_seq = GRU(n_hidden_units, activation = 'tanh',
                      return_sequences=True)(dec_emb, initial_state = h_enc)
    O_h = TimeDistributed(Dense(2*n_hidden_units, activation = 'linear'))(decoder_seq)
    O_y = TimeDistributed(Dense(2*n_hidden_units, activation = 'linear'))(dec_emb)
    O_c = Dense(2*n_hidden_units, activation = 'linear')(h_enc)
    O_m = Dense(2*n_hidden_units, activation = 'linear')(m_x)
    # max_out activation
    s = Add()([O_h, O_y, O_c, O_m])
    s = Reshape([max_len, 2*n_hidden_units, 1])(s)
    s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s)
    s = Reshape([max_len, n_hidden_units])(s)
    # outputs are probability Distribution of next word given previous tokens and input sentence tokens
    outputs = TimeDistributed(Dense(K, activation = 'softmax'))(s)
    
    model = Model(inputs = [encoder_input, decoder_input], outputs= outputs)
    model.summary()
    return model
