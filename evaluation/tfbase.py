
# base
from .binning import continuous_prediction
from .metrics import unadjusted_mse, unadjusted_mae
from .base import EvaluationModel

# tf imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import numpy as np

class KerasEvaluationModel(EvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.underlyingKerasModel = None
	
	def train(self, x, y, pivots, optimizer, epochs_max, validation_x, validation_y, *extraargs, **extrakwargs):
		self.underlyingKerasModel.compile(
			optimizer=optimizer,
			loss="mse",
			metrics=[ "mae" ],
			loss_weights=None,
			weighted_metrics=None,
			run_eagerly=True,
			steps_per_execution=None
		)
		
		earlyStoppingCallback = EarlyStopping(
			monitor='val_loss',
			patience=10,
			verbose=0,
			min_delta=0,
			mode='min',
			restore_best_weights=True,
			baseline=None,
		)
		# modelCheckpointSaveCallback = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
		reduceLearningRateOnPlateauCallback = ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.4,
			patience=5,
			verbose=0,
			# min_delta=1e-4,
			mode='min'
		)

		# train within the training radius
		history = self.underlyingKerasModel.fit(
			x=x, y=y,
			batch_size=20, epochs=epochs_max,
			callbacks=[earlyStoppingCallback, reduceLearningRateOnPlateauCallback],
			validation_split=0.20,
			verbose=0
		)

		validation_losses = history.history['val_loss']
		best_epoch_n = np.argmin(validation_losses)

		return best_epoch_n

class KerasFeedforwardEvaluationModel(KerasEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)

	def prepare_evaluation_data(self, x, y, train_y, y_normalizer):
		model = self.underlyingKerasModel

		variant_evaluation_predictions = model.predict(x).flatten()

		eval_result = model.evaluate(
			x=x, y=y,
			batch_size=None, verbose=0, sample_weight=None, steps=None,
			callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
			return_dict=True
		)

		eval_mse = eval_result['loss']
		eval_mae = eval_result['mae']

		return x, y, variant_evaluation_predictions, eval_mse, eval_mae

class KerasRecurrentEvaluationModel(KerasEvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.sequence_length = args.sequence_length

	def prepare_evaluation_data(self, x, y, train_y, y_normalizer):
		model = self.underlyingKerasModel

		variant_evaluation_x = x
		variant_evaluation_y = y
		future_size = variant_evaluation_y.size-1 # -1 for the endpoint, as we will be taking the midpoints of bin intervals
		variant_evaluation_predictions = continuous_prediction(model=model, past=train_y[-self.sequence_length:], window_length=self.sequence_length, future_length=future_size)
		variant_evaluation_predictions = variant_evaluation_predictions[self.sequence_length:]

		variant_evaluation_x = (variant_evaluation_x[1:] + variant_evaluation_x[:-1]) / 2
		variant_evaluation_y = variant_evaluation_y[:-1]

		mse = unadjusted_mse(
			y_true=variant_evaluation_y,
			y_pred=variant_evaluation_predictions,
		)
		mae = unadjusted_mae(
			y_true=variant_evaluation_y,
			y_pred=variant_evaluation_predictions,
		)

		return variant_evaluation_x, variant_evaluation_y, variant_evaluation_predictions, mse, mae
