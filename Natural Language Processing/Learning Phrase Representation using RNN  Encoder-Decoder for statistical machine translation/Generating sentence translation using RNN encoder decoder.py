import tensorflow as tf 
from tensorflow import keras
from keras.layers import *
from keras.applications import VGG19

# 1. Encoder Decoder neural network for parameter training.
def Encoder_decoder_trainer(max_len, K, latent_dim = 1000, dim_embed = 100):
        
        # source input sentence -> encoder -> h_enc, m_x 
    encoder_input = Input(shape = (max_len,), name = 'encoder_input')
    enc_emb = Embedding(input_dim = K, output_dim = dim_embed, name = 'encoder_embedding')(encoder_input)
    m_x = GlobalAveragePooling1D(name = 'm_x')(enc_emb)
    h_enc = GRU(latent_dim, activation = 'tanh', name = 'encoder_GRU')(enc_emb) # returns only last step output (dimension of n_hidden_units
        
        # target input sentence, h_enc, m_x -> decoder -> output probability distribution <- target sentence
        # decoder input sentence are lagged.
    decoder_input = Input(shape = (max_len,), name = 'decoder_input')
    dec_emb = Embedding(input_dim = K, output_dim = dim_embed, name = 'decoder_embedding')(decoder_input)
    s_dec = GRU(latent_dim, activation = 'tanh', return_sequences=True, name = 'decoder_GRU')(dec_emb, initial_state = h_enc) # returns outputs of each step (dim of max_len * n_hidden_units)
        #create vector length of 2*n_hidden_units on each time step, to apply maxout activation
    O_h = TimeDistributed(Dense(2*latent_dim, activation = 'linear', name = 'O_h'))(s_dec) 
    O_y = TimeDistributed(Dense(2*latent_dim, activation = 'linear', name = 'O_y'))(dec_emb)
    O_c = Dense(2*latent_dim, activation = 'linear', name = 'O_c')(h_enc)
    O_m = Dense(2*latent_dim, activation = 'linear', name = 'O_m')(m_x)
    s = Add()([O_h, O_y, O_c, O_m]) # add them all together
    s = Reshape([max_len, 2*latent_dim, 1])(s)
    s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s) # max-out activation
    s = Reshape([max_len, latent_dim])(s) # dimension of max_len * n_hidden_units
    output = TimeDistributed(Dense(K, activation = 'softmax'))(s) #dimension of max_len * K(vocab_size) -> probability distribution of each words
    
    return Model(inputs = [encoder_input, decoder_input], outputs = output)

# 2. Getting trained layers from trainer model
encoder_embedding_layer = trainer.layers[2]
decoder_embedding_layer = trainer.layers[3]
encoder_GRU = trainer.layers[4]
decoder_GRU = trainer.layers[5]
O_h_dense = trainer.layers[7]
O_y_dense = trainer.layers[8]
O_c_dense = trainer.layers[9]
O_m_dense = trainer.layers[10]
output_dense = trainer.layers[15]

# Config
latent_dim = 1000
dim_embed = 100
max_len = 30

# 3. Build source sentence encoder
input = Input(shape = (max_len,))
enc_emb = encoder_embedding_layer(input)
h_enc = encoder_GRU(enc_emb)
m_x = GlobalAveragePooling1D()(enc_emb)

infer_encoder = Model(inputs = input, outputs = [h_enc, m_x])

# 4. Build sentence decoder
h_enc = Input(shape = (latent_dim,))
m_x = Input(shape = (dim_embed,))
decoder_input = Input(shape = (1,))
decoder_initial_state = Input(shape=(latent_dim,)) #will be 
dec_emb = decoder_embedding_layer(decoder_input)
hidden_state = decoder_GRU(dec_emb, initial_state = decoder_initial_state)
O_h = O_h_dense(hidden_state)
O_y = O_y_dense(dec_emb)
O_c = O_c_dense(h_enc)
O_m = O_m_dense(m_x)
s = Add()([O_h, O_y, O_c, O_m]) # add them all together
s = Reshape([1, 2*latent_dim, 1])(s)
s = MaxPool2D(pool_size=(1,2), strides=(1,2))(s) # max-out activation
s = Reshape([1, latent_dim])(s) # dimension of max_len * n_hidden_units
output = output_dense(s)

infer_decoder = Model(inputs = [h_enc, m_x, decoder_input, decoder_initial_state], outputs = [output, hidden_state])

# 5. sentence generator based on encoder and decoder
def generate_sentence(source_sentence, encoder, decoder, max_length):
    sample_tokens = [0]
    length = 0
    h_enc, m_x = encoder.predict(source_sentence)
    terminate_condition = False
    while terminate_condition == False:
        if len(sample_tokens) == 1:
            output, hs = decoder.predict([h_enc, m_x, np.array(sample_tokens[-1]), h_enc])
        else:
            output, hs = decoder.predict([h_enc, m_x, np.array(sample_tokens[-1]), hs])
        next_token = np.argmax(output)
        sample_tokens.append(next_token)
        length += 1

        if sample_tokens[-1] == 0 or length > max_length:
            terminate_condition = True

    return sample_tokens
