import numpy as np
import sys
import time
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
from keras.applications import vgg16, imagenet_utils
import keras.backend as K
import numpy
import math
import matplotlib.pyplot as plt


from keras.applications import VGG19
model = VGG19()

#output feature map of each block
block_1 = model.layers[3].output
block_2 = model.layers[6].output
block_3 = model.layers[11].output
block_4 = model.layers[16].output
block_5 = model.layers[21].output

# reconstructing layers_4 to input size
block_4 = keras.layers.UpSampling2D(size = (2,2))(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_4_4 = keras.layers.Conv2DTranspose(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_4_4(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_4_3 = keras.layers.Conv2DTranspose(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_4_3(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_4_2 = keras.layers.Conv2DTranspose(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_4_2(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_4_1 = keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_4_1(block_4)

block_4 = keras.layers.UpSampling2D(size = (2,2))(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_3_4 = keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_3_4(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_3_3 = keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_3_3(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_3_2 = keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_3_2(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_3_1 = keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_3_1(block_4)

block_4 = keras.layers.UpSampling2D(size = (2,2))(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_2_2 = keras.layers.Conv2DTranspose(filters= 128, kernel_size = (3,3), strides= 1, padding = 'same')
block_4 = deconv_2_2(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_2_1 = keras.layers.Conv2DTranspose(filters= 64, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_2_1(block_4)

block_4 = keras.layers.UpSampling2D(size = (2,2))(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_1_2 = keras.layers.Conv2DTranspose(filters= 64, kernel_size = (3,3), strides= 1, padding = 'same')
block_4 = deconv_1_2(block_4)
block_4 = keras.layers.Activation('relu')(block_4)
deconv_1_1 = keras.layers.Conv2DTranspose(filters= 3, kernel_size = (3,3), strides = 1, padding = 'same')
block_4 = deconv_1_1(block_4)

# reconstructing layers 3 to input size
block_3 = keras.layers.UpSampling2D(size = (2,2))(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_3_4(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_3_3(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_3_2(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_3_1(block_3)

block_3 = keras.layers.UpSampling2D(size = (2,2))(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_2_2(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_2_1(block_3)

block_3 = keras.layers.UpSampling2D(size = (2,2))(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_1_2(block_3)
block_3 = keras.layers.Activation('relu')(block_3)
block_3 = deconv_1_1(block_3)

# reconstructing layers 2 to input size
block_2 = keras.layers.UpSampling2D(size = (2,2))(block_2)
block_2 = keras.layers.Activation('relu')(block_2)
block_2 = deconv_2_2(block_2)
block_2 = keras.layers.Activation('relu')(block_2)
block_2 = deconv_2_1(block_2)

block_2 = keras.layers.UpSampling2D(size = (2,2))(block_2)
block_2 = keras.layers.Activation('relu')(block_2)
block_2 = deconv_1_2(block_2)
block_2 = keras.layers.Activation('relu')(block_2)
block_2 = deconv_1_1(block_2)

# reconstructing layers 1 to input size
block_1 = keras.layers.UpSampling2D(size = (2,2))(block_1)
block_1 = keras.layers.Activation('relu')(block_1)
block_1 = deconv_1_2(block_1)
block_1 = keras.layers.Activation('relu')(block_1)
block_1 = deconv_1_1(block_1)


model_projection = Model(model.input, [block_1, block_2, block_3, block_4])
