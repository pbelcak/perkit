# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from typeguard import typechecked
from typing import Tuple
import numpy as np
import itertools
import math
from skopt import gp_minimize


# base
from ..base import EvaluationModel

# local imports
from .lifecycle import *
from .units import *

class BayesModel(EvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.best_pop = None

	def train(self, x, y, pivots, optimizer, epochs_max, validation_x, validation_y, max_generations: int, initial_population_size: int = 8, root_population_size: int = 3, children_population_size: int = 6):
		self.optimizer = optimizer

		pops = {}

		def f(period_guess):
			period_guess = period_guess[0]
			pop = self.makeModel(generation=0, identifier=-1, init_period=period_guess, root_identifier=-1)
			train_data, train_y = x, y
			es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
			history = pop.fit(train_data, train_y, epochs=epochs_max, batch_size=20, verbose=0, callbacks=[es_callback])
			last_loss = history.history['loss'][-1]
			pops[period_guess] = pop

			return last_loss

		res = gp_minimize(f,                  # the function to minimize
                  [(0.5, 1.0)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=max_generations*children_population_size,         # the number of evaluations of f
                  n_random_starts=initial_population_size,  # the number of random initialization points
				)   # the random seed

		print("x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun))
		
		self.best_pop = pops[res.x[0]]

	def prepare_evaluation_data(self, x, y, train_y, y_normalizer):
		model = self.best_pop

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

	def makeModel(self, generation: int, identifier: int, init_period: float, ancestors: Tuple[keras.Model, keras.Model] = None, root_identifier: int = -1):
		theFitterAncestor: keras.Model = None
		theLessFitAncestor: keras.Model = None
		if ancestors is not None:
			if ancestors[0] is None:
				theFitterAncestor = ancestors[1]
			elif ancestors[1] is None:
				theFitterAncestor = ancestors[0]
			else:
				theFitterAncestor, theLessFitAncestor = (ancestors[0], ancestors[1]) if ancestors[0].unfitness <= ancestors[1].unfitness else (ancestors[1], ancestors[0])

		# input part
		inputs = keras.Input(shape=(1,), name="number")
		
		# genetic layer part
		geneticLayer, afterGeneticLayer = None, None
		if theFitterAncestor is None:
			geneticLayer = GeneticUnit(period=init_period, name="genetic_unit")
			afterGeneticLayer = geneticLayer(inputs)
		else:
			geneticLayer, afterGeneticLayer = keras_clone_layer(theFitterAncestor.get_layer(name="genetic_unit"), inputs=inputs)
			geneticLayer.period = init_period
			
			if theLessFitAncestor is not None:
				theFitterAncestor.mated_with.append(theLessFitAncestor.name)
				theLessFitAncestor.mated_with.append(theFitterAncestor.name)
		
		# output part
		outputs = afterGeneticLayer

		model = keras.Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=self.optimizer, loss="mse", metrics=["mse", "mae"])
		
		model.root_identifier = root_identifier
		model.parentName = None if theFitterAncestor is None else theFitterAncestor.name
		model.birth = generation
		
		model.fits = 0
		model.unfitness = -1
		model.mated_with = []
		
		return model
