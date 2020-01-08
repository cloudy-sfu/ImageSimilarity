import keras
import cv2
import os
from random import random
import numpy as np

def conv_output(model, layer_name, img):
    """
    Get the output of conv layer.
    :param: model: keras model.
    :param: layer_name: name of layer in the model.
    :param: img: processed input image. (in batch)
    :return: intermediate_output: feature map.
    """
    input_img = model.input  # placeholder for the input images
    try:
        out_conv = model.get_layer(layer_name).output  # placeholder for the conv output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))
    intermediate_layer_model = keras.layers.Model(inputs=input_img, outputs=out_conv)  # get the intermediate layer model    
    intermediate_output = intermediate_layer_model.predict(img)  # get the output of intermediate layer model
    return intermediate_output[0]
    