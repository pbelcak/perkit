# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from typeguard import typechecked
from tensorflow_addons.activations.snake import snake
from tensorflow_addons.utils import types

# local imports
from ..tfbase import KerasFeedforwardEvaluationModel

class SnakeModel(KerasFeedforwardEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = SnakeKerasModel(width=width)

class TSnakeModel(KerasFeedforwardEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = SnakeKerasModel(width=width, train_frequency=True)

class SnakeKerasModel(keras.Model):
	def __init__(self, width: int = 1, frequency_initializer: types.Initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.99, seed=None), train_frequency: bool = False, **kwargs):
		super().__init__(**kwargs)
		self.neurons = [SnakeNeuron(frequency_initializer=frequency_initializer, train_frequency=train_frequency) for i in range(0, width)]
		self.linear = keras.layers.Dense(1, activation="linear")

	def call(self, inputs):
		# inputs = inputs[:, tf.newaxis] LOCALHOST ONLY
		x = keras.layers.concatenate([neuron(inputs) for neuron in self.neurons], axis=1)
		# x = keras.layers.Flatten()(x)
		x = self.linear(x)
		return x

def makeSnakeModel(width: int = 1, frequency_initializer: types.Initializer = "ones", train_frequency: bool = False, **kwargs):
	inputs = keras.layers.Input(shape=(20,1))
	neurons = keras.layers.concatenate([SnakeNeuron(frequency_initializer=frequency_initializer, train_frequency=train_frequency)(inputs) for i in range(0, width)], axis=1)
	outputs = keras.layers.Dense(1, activation="linear")(neurons)

	return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)

class SnakeNeuron(tf.keras.layers.Layer):
	@typechecked
	def __init__(self, frequency_initializer: types.Initializer = "ones", train_frequency: bool = False, **kwargs):
		super().__init__(**kwargs)
		self.train_frequency = train_frequency
		self.frequency_initializer = tf.keras.initializers.get(frequency_initializer)
		self.frequency = self.add_weight(
			initializer=self.frequency_initializer,
			trainable=train_frequency
		)

	def call(self, inputs):
		return snake(x=inputs, frequency=self.frequency)

	def get_config(self):
		config = {
			"frequency_initializer": tf.keras.initializers.serialize(
				self.frequency_initializer
			),
			"train_frequency": str(self.train_frequency),
		}
		base_config = super().get_config()
		return {**base_config, **config}