import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, layer_before_softmax, pred_index=None):
    '''
    img_array : (1, width, height, 3) dimensional tensor
    model : CNN model
    last_conv_layer_name : name of last convolutional layer
    layer_before_softmax : flattend layer before softmax
    pred_index : prediction indices
    '''
    
    # model that outputs last_conv_activation & final_prediction, given input image tensor
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(layer_before_softmax).output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # 
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # dimension 0 is batch_size -> eliminate the dimension
    grads = grads[0]
    last_conv_layer_output = last_conv_layer_output[0]
    # GAP on grads -> weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    grad_cam = np.zeros(dtype=np.float32, shape = last_conv_layer_output.shape[0:2])
    # linear combination with weights of pooled_grads
    for i, a in enumerate(pooled_grads):
        grad_cam += a * last_conv_layer_output[:,:,i]
    grad_cam = grad_cam.numpy()
    # upsampling using bilinear interpolation
    grad_cam = cv2.resize(grad_cam, (224, 224))
    # ReLu activation
    grad_cam = np.maximum(grad_cam, 0)
    # Normalization
    grad_cam = grad_cam / grad_cam.max()
    
    return grad_cam
