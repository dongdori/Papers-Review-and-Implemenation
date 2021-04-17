import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import *


class RNN_encoder_decoder():
    def __init__(self, K, max_len, n_hidden_units = 1000, dim_embed = 100):
        self.K = K
        self.max_len = max_len
        self.n_hidden_units = n_hidden_units
        self.dim_embed = dim_embed
    
    def Encoder_decoder_trainer(self):
        #encoding layers
        self.encoder_input = None
        self.encoder_Embedding = None
        self.encoder_GRU = None
        #decoding layers
        self.decoder_input = None
        self.decoder_Embedding = None
        self.decoder_GRU = None
        self.O_h = None
        self.O_y = None
        self.O_c = None
        self.O_m = None
        self.s = None
        self.outputs = None
        
        self.encoder_input = Input(shape = (self.max_len,))
        self.encoder_Embedding = Embedding(input_dim = self.K, output_dim = self.dim_embed, input_length = self.max_len)
        enc_emb = self.encoder_Embedding(self.encoder_input)
        m_x = GlobalAveragePooling1D()(enc_emb)
        self.encoder_GRU = GRU(self.n_hidden_units, activation = 'tanh')
        h_enc = self.encoder_GRU(enc_emb)
        
        self.decoder_input = Input(shape = (self.max_len,))
        self.decoder_Embedding = Embedding(input_dim = self.K, output_dim = self.dim_embed, input_length = self.max_len)
        dec_emb = self.decoder_Embedding(self.decoder_input)
        self.decoder_GRU = GRU(self.n_hidden_units, activation = 'tanh', return_sequences=True, return_state=True)
        s_dec, _ = self.decoder_GRU(dec_emb, initial_state = h_enc)
        self.O_h = TimeDistributed(Dense(2*self.n_hidden_units, activation = 'linear'))
        O_h = self.O_h(s_dec)
        self.O_y = TimeDistributed(Dense(2*self.n_hidden_units, activation = 'linear'))
        O_y = self.O_y(dec_emb)
        self.O_c = Dense(2*self.n_hidden_units, activation = 'linear')
        O_c = self.O_c(h_enc)
        self.O_m = Dense(2*self.n_hidden_units, activation = 'linear')
        O_m = self.O_m(m_x)
        s = Add()([O_h, O_y, O_c, O_m])
        s = Reshape([self.max_len, 2*self.n_hidden_units, 1])(s)
        s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s)
        s = Reshape([self.max_len, self.n_hidden_units])(s)
        self.output_layer = TimeDistributed(Dense(self.K, activation = 'softmax'))
        output = self.output_layer(s)
        return Model(inputs = [self.encoder_input, self.decoder_input], outputs = output)
    
    def Encoder_decoder_predict(self):
        #encoding input
        encoder_input = self.encoder_input
        enc_emb = self.encoder_Embedding(encoder_input)
        m_x = GlobalAveragePooling1D()(enc_emb)
        h_enc = self.encoder_GRU(enc_emb)
        #decoding sequentially
        decoder_input = self.decoder_input
        dec_emb = self.decoder_Embedding(decoder_input)
        s_dec, hidden_state = self.decoder_GRU(dec_emb, initial_state = h_enc)
        O_h = self.O_h(s_dec)
        O_y = self.O_y(dec_emb)
        O_m = self.O_m(m_x)
        O_c = self.O_c(h_enc)
        s = Add()([O_h, O_y, O_m, O_c])
        s = Reshape([self.max_len, 2*self.n_hidden_units, 1])(s)
        s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s)
        s = Reshape([self.max_len, self.n_hidden_units])(s)
        output = self.output_layer(s)
        
        return Model(inputs = [encoder_input, decoder_input], outputs = [output, hidden_state])
        
