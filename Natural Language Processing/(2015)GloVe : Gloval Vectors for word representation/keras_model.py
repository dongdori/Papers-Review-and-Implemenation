from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# hyperparameters
x_max = 100
alpha = 3.0 / 4.0

class glove_model(Model):
    def __init__(self, embed_dim):
        self.W_target = Embedding(V, embed_dim, input_length = 1, name = 'central_target_embedding')
        self.b_target = Embedding(V, 1, input_length = 1, name = 'bias_target_embedding')
        self.W_context = Embedding(V, embed_dim, input_length = 1, name = 'central_context_embedding')
        self.b_context = Embedding(V, 1, input_length = 1, name = 'bias_context_embedding') 
        self.dot = Dot(axes=-1)
    def forward(self, context_input, target_input):
        W_t = self.W_target(target_input)
        b_t = self.b_target(target_input)
        W_c = self.W_context(context_input)
        b_c = self.b_context(context_input)
        
        x = self.dot([W_t, W_c])
        x = x.reshape((1,))
        b_c = b_c.reshape((1,))
        b_t = b_t.reshape((1,))
        
        output = Add()([x, b_c, b_t])
        return output
  
# cost function
def glove_loss(y_true, y_pred):
    f = K.pow(K.clip(y_true / x_max, 0.0, 1.0), alpha) # 0 <= y_true/ x_max <= 1
    return K.sum(f * K.square(y_pred - K.log(y_true)), axis=-1)
