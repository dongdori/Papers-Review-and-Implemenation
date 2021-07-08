import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# Token Embedding + Sinusodial positional embedding Block
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.token_emb = Embedding(input_dim = vocab_size, output_dim = d_model)
    def build(self, input_shape):
        # build sinusoid positional encodings
        encodings = np.zeros((self.maxlen, self.d_model))
        for pos in range(self.maxlen):
            for i in range(self.d_model):
                if i % 2 == 0:
                    encodings[pos, i] = np.sin(pos / 10000**(2*i/self.maxlen))
                else:
                    encodings[pos, i] = np.cos(pos / 10000**(2*i/self.maxlen))
        self.pos_encoding = tf.Variable(initial_value = encodings, dtype = np.float32, trainable = False)
    def call(self, x):
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
    def call(self, inputs):
        # multihead attention & skip connection
        out1 = self.multiheadattention(query = inputs, value = inputs, key = inputs)
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
    def call(self, inputs, enc_output, mask):
        # masked multihead attention layer
        out1 = self.mha1(query = inputs, value = inputs, key = inputs, attention_mask = mask)
        out1 = self.layernorm1(inputs + out1)
        # multihead attention layer
        out2 = self.mha2(query = enc_output, value = enc_output, key = out1)
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
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, d_model)
    def call(self, inputs):
        # token embedding + positional embedding
        x = self.embedding(inputs)
        for encoderlayer in self.encoderlayers:
            x = encoderlayer(x)
        return x    
# Decoder with n_layers
class Decoder(Layer):
    def __init__(self, n_layers, maxlen, num_heads, d_v, d_model, vocab_size, fnn):
        super(Decoder, self).__init__()
        self.decoderlayers = [DecoderBlock(maxlen, num_heads, d_v, d_model, fnn) for _ in range(n_layers)]
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
    def call(self, inputs, enc_output, mask):
        x = self.embedding(inputs)
        for decoderlayer in self.decoderlayers:
            x = decoderlayer(x, enc_output, mask)
        return x
