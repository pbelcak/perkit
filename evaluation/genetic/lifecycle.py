# tf and tf-related imports
from tensorflow import keras

def keras_clone_layer(layer_to_clone: keras.layers.Layer, inputs):
    config = layer_to_clone.get_config()
    weights = layer_to_clone.get_weights()
    cloned_layer = type(layer_to_clone).from_config(config)
    pipelined = cloned_layer(inputs)
    cloned_layer.set_weights(weights)
    
    return cloned_layer, pipelined

