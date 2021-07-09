import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# utils
## 1. Functions for positional encoding(sinusodial)
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

## 2. Functions for creating masks
def create_look_back_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  # decoder padding mask applied on encoder output(2nd multihead attention layer)
  dec_padding_mask = create_padding_mask(inp)
  # decoder masking on 1st multihead attention layer
  look_back_mask = create_look_back_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  # if token is [PAD] or future token -> masking
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


# Token Embedding + Sinusodial positional embedding Block
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.token_emb = Embedding(input_dim = vocab_size, output_dim = d_model)
        self.pos_encoding = positional_encoding(self.maxlen, self.d_model)
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
    def call(self, inputs, padding_mask):
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
# Encoder with n_layers (TokenPostionalEmbedding + Encoder Blocks)
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
# Decoder with n_layers (TokenPositionalEmbedding + Decoder Blocks)
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

# Transformer (Encoder + Decoder + output layer)    
class TransFormer(Model):
    def __init__(self, n_layers, num_heads, d_v, d_model, tar_vocab_size, src_vocab_size, tar_maxlen, src_maxlen, fnn):
        super(TransFormer, self).__init__()
        self.Encoder = Encoder(n_layers, src_maxlen, num_heads, d_v, d_model, src_vocab_size, fnn)
        self.Decoder = Decoder(n_layers, tar_maxlen, num_heads, d_v, d_model, tar_vocab_size, fnn)
        self.output_layer = Dense(tar_vocab_size, activation = 'softmax')
    def call(self, input_src, input_tar, look_back_mask, enc_padding_mask, dec_padding_mask):
        enc_output = self.Encoder(input_src, enc_padding_mask)
        dec_output = self.Decoder(input_tar, enc_output, dec_padding_mask, look_ahead_mask)
        output = self.output_layer(dec_output)
        return output
