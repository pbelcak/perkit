# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from typeguard import typechecked
from typing import Tuple
import numpy as np
import itertools
import math

# base
from ..base import EvaluationModel

# local imports
from .lifecycle import *
from .units import *

class GeneticModel(EvaluationModel):
	def __init__(self, width, args):
		super().__init__(width, args)
		self.best_pop = None

	def train(self, x, y, pivots, optimizer, epochs_max, validation_x, validation_y, max_generations: int, initial_population_size: int = 8, root_population_size: int = 3, children_population_size: int = 6):
		self.optimizer = optimizer

		population = self.initPopulation(size=initial_population_size, min_period=0.5, max_period=1.0)
		population_to_reproduce = [*population]

		for generation in range(1, max_generations+1):
			population, population_to_reproduce = self.evolve(population=population, population_to_reproduce=population_to_reproduce, generation=generation, train_pair=(x, y), fitness_pair=(validation_x, validation_y),
			min_roots=root_population_size, n_bp=children_population_size, unfitness_relative_offset=0.05, epochs=epochs_max)
			periods = [pop.get_layer(name='genetic_unit').period for pop in population_to_reproduce]
			best_fitness = population_to_reproduce[0].unfitness
			# print(f"Best fitness: {best_fitness}")
		
		population = sorted(population, key=lambda pop: pop.unfitness)
		self.best_pop = population[0]

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

	def initPopulation(self, size: int, min_period: float, max_period: float):
		if size < 2:
			raise Exception("The size is too small!")
		
		g0_step: float = (max_period - min_period) / (size - 1)
		g0_inits = np.arange(min_period, max_period+g0_step, g0_step) # there will be a size of these
		g1_inits = np.arange(min_period+g0_step/2, max_period, g0_step) # and there will be a size-1 of these
		
		population = list(itertools.chain(
			( self.makeModel(generation=0, identifier=i, init_period=period, root_identifier=i) for period, i in zip(g0_inits, range(0, len(g0_inits))) ),
			( self.makeModel(generation=1, identifier=j, init_period=period) for period, j in zip(g1_inits, range(len(g0_inits), len(g0_inits)+len(g1_inits))) ),
		))
			
		return population

	def reproduce(self, ancestors: Tuple[keras.Model, keras.Model], generation: int, identifier: int):
		ancestor_left_gu = ancestors[0].get_layer(name="genetic_unit")
		ancestor_right_gu = ancestors[1].get_layer(name="genetic_unit")
		
		deltas = [(ancestor_right_gu.period - ancestor_left_gu.period) / 2]

		for delta in deltas:
			period = ancestor_left_gu.period + delta
			child = self.makeModel(generation=generation, identifier=identifier, init_period=period, ancestors=ancestors)
			
			yield child
