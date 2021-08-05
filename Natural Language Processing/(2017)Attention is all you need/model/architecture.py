import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

from padding_func import create_masks, create_padding_masks, create_lookahead_masks

# Embedding Block
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.token_emb = Embedding(input_dim = vocab_size, output_dim = d_model)
        self.pos_encoding = positional_encoding(self.maxlen, self.d_model)
    def __call__(self, x):
        token_emb = self.token_emb(x)
        return self.pos_encoding + token_emb
 
# Encoder Block
class EncoderBlock(Layer):
    def __init__(self, maxlen, num_heads, d_v, d_model, fnn):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.d_v = d_v
        self.d_model = d_model
        self.fnn = fnn

        self.multiheadattention = MultiHeadAttention(num_heads = self.num_heads,
                                                     key_dim = self.d_v,
                                                     dropout = 0.1)
        self.positionalfeedforward_1 = Conv1D(filters = fnn, kernel_size = 1, activation = 'relu')
        self.positionalfeedforward_2 = Conv1D(filters = d_model, kernel_size = 1, activation = 'linear')
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
    def __call__(self, inputs, padding_mask):
        # multihead attention & skip connection
        out1 = self.multiheadattention(query = inputs, value = inputs, key = inputs, attention_mask = padding_mask)
        out1 = self.layernorm_1(inputs + out1)
        # positional feedforward 
        out2 = self.positionalfeedforward_1(out1)
        out2 = Reshape([self.maxlen, 2048])(out2)
        out2 = self.positionalfeedforward_2(out2)
        out2 = Reshape([self.maxlen, self.d_model])(out2)
        output = self.layernorm_2(out1 + out2)

        return output
# Decoder Block
class DecoderBlock(Layer):
    def __init__(self, maxlen, num_heads, d_v, d_model, fnn):
        super(DecoderBlock, self).__init__()
        self.maxlen = maxlen
        self.d_v = d_v
        self.fnn = fnn
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.mha1 = MultiHeadAttention(key_dim = self.d_model, num_heads = self.num_heads, dropout = 0.1)
        self.mha2 = MultiHeadAttention(key_dim = self.d_model, num_heads = self.num_heads, dropout = 0.1)
        self.positionalfeedforward_1 = Conv1D(filters = fnn, kernel_size = 1, activation = 'relu')
        self.positionalfeedforward_2 = Conv1D(filters = d_model, kernel_size = 1, activation = 'linear')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self, inputs, enc_output, padding_mask, look_back_mask):
        # masked multihead attention layer
        out1 = self.mha1(query = inputs, value = inputs, key = inputs, attention_mask = look_back_mask)
        out1 = self.layernorm1(inputs + out1)
        # multihead attention layer
        out2 = self.mha2(query = enc_output, value = enc_output, key = out1, attention_mask = padding_mask)
        out2 = self.layernorm2(out1 + out2)
        # positionwise feedforward layer
        out3 = self.positionalfeedforward_1(out2)
        out3 = Reshape([self.maxlen, self.fnn])(out3)
        out3 = self.positionalfeedforward_2(out3)
        out3 = Reshape([self.maxlen, self.d_model])(out3)
        output = self.layernorm_2(out1 + out2)

        return output
# Encoder with n_layers
class Encoder(Layer):
    def __init__(self, n_layers, maxlen, num_heads, d_v, d_model, vocab_size, fnn):
        super(Encoder, self).__init__()
        self.encoderlayers = [EncoderBlock(maxlen, num_heads, d_v, d_model, fnn) for _ in range(n_layers)]
        self.encoding = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
    def call(self, inputs, padding_mask):
        # token embedding + positional embedding
        x = self.encoding(inputs)
        for encoderlayer in self.encoderlayers:
            x = encoderlayer(x, padding_mask)
        return x        
      
# Decoder with n_layers
class Decoder(Layer):
    def __init__(self, n_layers, maxlen, num_heads, d_v, d_model, vocab_size, fnn):
        super(Decoder, self).__init__()
        self.decoderlayers = [DecoderBlock(maxlen, num_heads, d_v, d_model, fnn) for _ in range(n_layers)]
        self.encoding = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
    def call(self, inputs, enc_output, padding_mask, look_back_mask):
        x = self.encoding(inputs)
        for decoderlayer in self.decoderlayers:
            x = decoderlayer(x, enc_output, padding_mask, look_back_mask)
        return x
# put it all together
class TransFormer(Model):
    def __init__(self, config):
        super(TransFormer, self).__init__()
        self.n_layers = config['N_LAYERS']
        self.num_heads = config['NUM_HEADS']
        self.d_v = config['D_v']
        self.d_model = config['D_model']
        self.tar_vocab_size = config['tar_vocab_size']
        self.src_vocab_size = config['src_vocab_size']
        self.tar_maxlen = config['tar_MAXLEN'] 
        self.src_maxlen = config['src_MAXLEN']
        self.fnn = config['FNN']
        
        self.Encoder = Encoder(self.n_layers, self.src_maxlen, self.num_heads, self.d_v, self.d_model, self.src_vocab_size, self.fnn)
        self.Decoder = Decoder(self.n_layers, self.tar_maxlen, self.num_heads, self.d_v, self.d_model, self.tar_vocab_size, self.fnn)
        self.output_layer = Dense(self.tar_vocab_size, activation = 'softmax')
    def call(self, input_src, input_tar, look_back_mask, enc_padding_mask, dec_padding_mask):
        enc_output = self.Encoder(input_src, enc_padding_mask)
        dec_output = self.Decoder(input_tar, enc_output, dec_padding_mask, look_back_mask)
        output = self.output_layer(dec_output)
        return output
