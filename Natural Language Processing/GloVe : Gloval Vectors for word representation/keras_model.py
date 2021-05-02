from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# hyperparameters
x_max = 100
alpha = 3.0 / 4.0

# model
def glove_model(V, embed_dim):
    input_target = Input(shape = (1,))
    input_context = Input(shape = (1,))
    W_target_embedding = Embedding(V, embed_dim, input_length = 1, name = 'central_target_embedding')
    b_target_embedding = Embedding(V, 1, input_length = 1, name = 'bias_target_embedding')
    W_context_embedding = Embedding(V, embed_dim, input_length = 1, name = 'central_context_embedding')
    b_context_embedding = Embedding(V, 1, input_length = 1, name = 'bias_context_embedding')

    target_embedding = W_target_embedding(input_target)
    context_embedding = W_context_embedding(input_context)
    target_bias = b_target_embedding(input_target)
    context_bias = b_context_embedding(input_context)

    dot_product = Dot(axes=-1)([target_embedding, context_embedding])
    dot_product = Reshape((1, ))(dot_product)
    target_bias = Reshape((1,))(target_bias)
    context_bias = Reshape((1,))(context_bias)

    output = Add()([dot_product, bias_target, bias_context])

    return Model(inputs= [input_target, input_context], outputs = output)
  
# cost function
def glove_loss(y_true, y_pred):
    f = K.pow(K.clip(y_true / x_max, 0.0, 1.0), alpha) # 0 <= y_true/ x_max <= 1
    return K.sum(f * K.square(y_pred - K.log(y_true)), axis=-1)
