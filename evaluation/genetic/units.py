# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from typeguard import typechecked

class PeriodicUnit(tf.keras.layers.Layer):
    @typechecked
    def __init__(self, periodicLayer: keras.layers.Dense = None, composer: keras.layers.Dense = None,  **kwargs):
        super().__init__(**kwargs)
        
        if periodicLayer is None:
            self.periodicLayer = keras.layers.Dense(64, activation='relu')
        else:
            self.periodicLayer = periodicLayer
        
        if composer is None:
            self.composer = keras.layers.Dense(1, activation="linear")
        else:
            self.composer = composer
        
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        batch = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[1]
        
        periodProcessed = self.periodicLayer(inputs)
        output = self.composer(periodProcessed)
        
        return output

    def get_config(self):
        from keras.layers import serialize as serialize_layer  # pylint: disable=g-import-not-at-top
        
        base_config = super().get_config()
        return {
                **base_config,
                'periodicLayer': serialize_layer(self.periodicLayer),
                'composer': serialize_layer(self.composer),
        }
    
    @classmethod
    def from_config(cls, config):
        from keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
        
        periodicLayer = deserialize_layer(config.pop('periodicLayer'))
        composer = deserialize_layer(config.pop('composer'))
        return cls(periodicLayer=periodicLayer, composer=composer, **config)

class GeneticUnit(tf.keras.layers.Layer):
    @typechecked
    def __init__(self, period: float = 1.0, periodicUnit: PeriodicUnit = None, baseLayer: keras.layers.Layer = None, combinationLayer: keras.layers.Layer = None, **kwargs):
        super().__init__(**kwargs)
        
        # set the period
        self.period = period
        
        # deal with the periodic unit
        self.periodicUnit = periodicUnit if periodicUnit is not None else PeriodicUnit()
        
        # deal with base
        self.baseLayer = baseLayer if baseLayer is not None else keras.layers.Dense(4, activation='linear')
        
        # combine the periodic output with the base output
        self.combinationLayer = combinationLayer if combinationLayer is not None else keras.layers.Dense(1, activation='linear', use_bias = False)
        
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        batch = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[1]
        
        # periodic part
        moddedInputs = tf.math.floormod(inputs, self.period) # floormod will broadcast
        periodDecided = self.periodicUnit(moddedInputs)
        
        # base part
        baseProcessed = self.baseLayer(inputs)
        
        # the two put together
        output = self.combinationLayer(keras.layers.concatenate([ periodDecided, baseProcessed ]))
        
        return output

    def get_config(self):
        base_config = super().get_config()
        return {
                **base_config,
                'period': self.period,
                'periodicUnit': self.periodicUnit.get_config(),
                'baseLayer': self.baseLayer.get_config(),
                'combinationLayer': self.combinationLayer.get_config(),
        }
    
    @classmethod
    def from_config(cls, config):
        periodicUnit = PeriodicUnit.from_config(config.pop('periodicUnit'))
        baseLayer = keras.layers.Dense.from_config(config.pop('baseLayer'))
        combinationLayer = keras.layers.Dense.from_config(config.pop('combinationLayer'))
        return cls(periodicUnit=periodicUnit, baseLayer=baseLayer, combinationLayer=combinationLayer, **config)