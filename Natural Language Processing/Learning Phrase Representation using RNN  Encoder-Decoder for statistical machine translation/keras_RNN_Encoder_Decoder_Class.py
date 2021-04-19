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
        
        # source input sentence -> encoder -> h_enc, m_x 
        self.encoder_input = Input(shape = (self.max_len,))
        self.encoder_Embedding = Embedding(input_dim = self.K, output_dim = self.dim_embed, input_length = self.max_len)
        enc_emb = self.encoder_Embedding(self.encoder_input)
        m_x = GlobalAveragePooling1D()(enc_emb)
        self.encoder_GRU = GRU(self.n_hidden_units, activation = 'tanh') # returns only last step output (dimension of n_hidden_units)
        h_enc = self.encoder_GRU(enc_emb)
        
        # target input sentence, h_enc, m_x -> decoder -> output probability distribution <- target sentence
        # decoder input sentence are lagged.
        self.decoder_input = Input(shape = (self.max_len,))
        self.decoder_Embedding = Embedding(input_dim = self.K, output_dim = self.dim_embed, input_length = self.max_len)
        dec_emb = self.decoder_Embedding(self.decoder_input)
        self.decoder_GRU = GRU(self.n_hidden_units, activation = 'tanh', return_sequences=True, return_state=True) # returns outputs of each step (dim of max_len * n_hidden_units)
        s_dec, _ = self.decoder_GRU(dec_emb, initial_state = h_enc)
        #create vector length of 2*n_hidden_units on each time step, to apply maxout activation
        self.O_h = TimeDistributed(Dense(2*self.n_hidden_units, activation = 'linear')) 
        O_h = self.O_h(s_dec)
        self.O_y = TimeDistributed(Dense(2*self.n_hidden_units, activation = 'linear'))
        O_y = self.O_y(dec_emb)
        self.O_c = Dense(2*self.n_hidden_units, activation = 'linear')
        O_c = self.O_c(h_enc)
        self.O_m = Dense(2*self.n_hidden_units, activation = 'linear')
        O_m = self.O_m(m_x)
        s = Add()([O_h, O_y, O_c, O_m]) # add them all together
        s = Reshape([self.max_len, 2*self.n_hidden_units, 1])(s)
        s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s) # max-out activation
        s = Reshape([self.max_len, self.n_hidden_units])(s) # dimension of max_len * n_hidden_units
        self.output_layer = TimeDistributed(Dense(self.K, activation = 'softmax')) #dimension of max_len * K(vocab_size) -> probability distribution of each words
        output = self.output_layer(s)
        return Model(inputs = [self.encoder_input, self.decoder_input], outputs = output)
    
    def predict_next_word_model(self, hideen_state = None):
        self.output_tokens = [0] 
        encoder_input = Input(shape = (self.max_len,)) #source sentence(dimension of (max_len,))
        enc_emb = self.encoder_Embedding(encoder_input) # dimension of (max_len, dim_embed)
        m_x = GlobalAveragePooling1D()(enc_emb) # dimension of (, dim_embed)
        h_enc = self.encoder_GRU(enc_emb) # creates encoding vector of source sentence (dimension of (, n_hidden_units))
        #decoding sequentially
        decoder_input = Input(shape = (1,)) # single token(CLS token or previousely generated word token) - dimension of (1, 1)
        dec_emb = self.decoder_Embedding(decoder_input) # dimension of (1, dim_embed)
        # returns h_<i+1> given h_<i> and y_<i>.
        # h_<0> = h_enc, y_<0> = first decoder_input, which is CLS token
        # h_<i>, y_<i> are determined by previouslely generated word.
        # h_<i> is hidden_state of previous word input, y_<i> is embedding vector of previous word input
        if len(self.output_tokens) == 1:
            s_dec, hidden_state = self.decoder_GRU(dec_emb, initial_state = h_enc)
        else:
            s_dec, hidden_state = self.decoder_GRU(dec_emb, initial_state = hidden_state)
        O_h = self.O_h(s_dec)
        O_y = self.O_y(dec_emb)
        O_m = self.O_m(m_x)
        O_c = self.O_c(h_enc)
        s = Add()([O_h, O_y, O_m, O_c])
        s = Reshape([1, 2*self.n_hidden_units, 1])(s)
        s = MaxPool2D(pool_size=(1,2), strides=(1,2), name = 'max_out')(s)
        s = Reshape([1, self.n_hidden_units])(s)
        output = self.output_layer(s)
        
        # encoder_input : source sentence tokens
        # decoder_input : previously generated word token or CLS token
        # output : conditional probability distribution of next word, given source sentence and previously generated word
        # hidden_state : hidden state of previous step that needs to be feed into next call of decoder_GRU
        model = Model(inputs = [encoder_input, decoder_input], outputs = [output, hidden_state])
        return model
