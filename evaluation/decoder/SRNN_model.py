# tf and tf-related imports
from tensorflow import keras

# local imports
from ..tfbase import KerasRecurrentEvaluationModel

class SRNNModel(KerasRecurrentEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = SRNNKerasModel(width=width)

class SRNNKerasModel(keras.Model):
    def __init__(self, width: int = 1):
        super().__init__()
        self.recurrent_units = keras.layers.SimpleRNN(units=width, recurrent_dropout=0.5)
        self.linear = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.recurrent_units(inputs)
        x = self.linear(x)
        return x