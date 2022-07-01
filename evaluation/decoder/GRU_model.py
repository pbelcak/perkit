# tf and tf-related imports
from tensorflow import keras

# local imports
from ..tfbase import KerasRecurrentEvaluationModel

class GRUModel(KerasRecurrentEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = GRUKerasModel(width=width)

class GRUKerasModel(keras.Model):
    def __init__(self, width: int = 1):
        super().__init__()
        self.recurrent_units = keras.layers.GRU(units=width, recurrent_dropout=0.5)
        self.linear = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.recurrent_units(inputs)
        x = self.linear(x)
        return x