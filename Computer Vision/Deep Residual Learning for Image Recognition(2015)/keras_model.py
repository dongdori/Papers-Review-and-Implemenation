import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *

def shortcut_1(x, n_featuremaps):
    y = Conv2D(filters = n_featuremaps, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters = n_featuremaps, kernel_size = (3,3), strides = (1,1), padding = 'same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Add()([x, y])
    return y

def shortcut_2(x, n_featuremaps):
    x_1 = Conv2D(filters = n_featuremaps, kernel_size = (1,1), strides = (1,1))(x)
    y = Conv2D(filters = n_featuremaps, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters = n_featuremaps, kernel_size = (3,3), strides = (1,1), padding = 'same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Add()([x_1, y])
    return y
  
input = Input(shape = (224,224, 3))
x = Conv2D(filters = 64, kernel_size=(7,7), strides = (2,2), activation = 'relu')(input)
x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)

x = shortcut_1(x, n_featuremaps = 64)
x = shortcut_1(x, n_featuremaps = 64)
x = shortcut_1(x, n_featuremaps = 64)
x = MaxPool2D(pool_size = (2,2), strides = (2,2))(x)

x = shortcut_2(x, n_featuremaps = 128)
x = shortcut_1(x, n_featuremaps = 128)
x = shortcut_1(x, n_featuremaps = 128)
x = shortcut_1(x, n_featuremaps = 128)
x = MaxPool2D(pool_size = (2,2), strides = (2,2))(x)

x = shortcut_2(x, n_featuremaps = 256)
x = shortcut_1(x, n_featuremaps = 256)
x = shortcut_1(x, n_featuremaps = 256)
x = shortcut_1(x, n_featuremaps = 256)
x = shortcut_1(x, n_featuremaps = 256)
x = shortcut_1(x, n_featuremaps = 256)
x = MaxPool2D(pool_size = (2,2), strides = (2,2))(x)

x = shortcut_2(x, n_featuremaps = 512)
x = shortcut_1(x, n_featuremaps = 512)
x = shortcut_1(x, n_featuremaps = 512)

x = AveragePooling2D(pool_size = (2,2))(x)
x = Flatten()(x)
output = Dense(1000, activation = 'softmax')(x)

model = keras.models.Model(input, output)


model.compile(optimizer=keras.optimizers.SGD(learning_rate = 0.1), loss = 'categorical_crossentropy')
model.summary()
