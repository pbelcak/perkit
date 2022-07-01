# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.utils import types

# local imports
from ..tfbase import KerasFeedforwardEvaluationModel

class SinCosModel(KerasFeedforwardEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = SinCosKerasModel(width=width)

class SinCosKerasModel(keras.Model):
	def __init__(self, width: int = 1, frequency_initializer: types.Initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.99, seed=None), train_frequency: bool = False):
		super().__init__()
		self.neurons = [SinCosNeuron() for i in range(0, width)]
		self.linear = keras.layers.Dense(1, activation="linear")

	def call(self, inputs):
		# inputs = inputs[:, tf.newaxis] LOCALHOST ONLY
		x = keras.layers.concatenate([neuron(inputs) for neuron in self.neurons], axis=1)
		# x = keras.layers.Flatten()(x)
		x = self.linear(x)
		return x

class SinCosNeuron(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.sinScale = self.add_weight()
		self.cosScale = self.add_weight()

		self.sinLogitLayer = keras.layers.Dense(1, activation="linear")
		self.cosLogitLayer = keras.layers.Dense(1, activation="linear")

	def call(self, inputs):
		return self.sinScale*tf.math.sin(self.sinLogitLayer(inputs)) + self.cosScale*tf.math.cos(self.cosLogitLayer(inputs))

	def get_config(self):
		config = {
			
		}
		base_config = super().get_config()
		return {**base_config, **config}