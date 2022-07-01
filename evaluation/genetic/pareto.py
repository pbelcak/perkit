# tf and tf-related imports
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import numpy as np
import itertools
import math

# base
from .base import GeneticModel

# local imports
from .lifecycle import *

class ParetoModel(GeneticModel):
	def evolve(self, population, population_to_reproduce, generation: int , train_pair: tuple, fitness_pair: tuple, min_roots: int, n_bp: int, unfitness_relative_offset: float, epochs: int):
		train_data, train_y = train_pair[0], train_pair[1]
		fitness_data, fitness_y = fitness_pair[0], fitness_pair[1]
		# print(f"Beggining the evolutionary cycle, population: {len(population)}, population to reproduce: {len(population_to_reproduce)}")
		
		# 1 reproduce (if not in the first or second generation)
		if generation > 1:
			population = sorted(population, key=lambda pop: pop.get_layer(name='genetic_unit').period)
			
			new_population = [*population]
			offspring_count = 0
			for pop_left, pop_right in zip(population, population[1:]):
				if pop_left in population_to_reproduce or pop_right in population_to_reproduce:
					if pop_left.name in pop_right.mated_with or pop_right.name in pop_left.mated_with:
						continue
					
					for child in self.reproduce(ancestors=(pop_left, pop_right), generation=generation, identifier=offspring_count):
						new_population.append(child)
						offspring_count += 1
			population = new_population
		
		# 2 face the environment and fitness test
		one_tenth: float = len(population) / 10.0
		for pop, pop_i in zip(population, range(0, len(population))):
			# if pop_i % one_tenth < 1:
			# 	print(f"\r - Training population {(pop_i // one_tenth)*10 : .0f}%")
			
			if pop.fits < 1:
				es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
				pop.fit(train_data, train_y, epochs=epochs, batch_size=20, verbose=0, callbacks=[es_callback])
				pop.fits += 1
				pop.unfitness = pop.evaluate(
					x=fitness_data, y=fitness_y, batch_size=20, sample_weight=None, steps=None,
					callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
					return_dict=True, verbose=0
				)['mse']
		
		# 3 sort by fitness
		max_unfitness = max(pop.unfitness for pop in population)
		min_unfitness = min(pop.unfitness for pop in population)
		unfitness_range = max_unfitness-min_unfitness
		# print(f"Min-Max unfitness: {min_unfitness}-{max_unfitness}")
		if max_unfitness <= 0:
			candidate_population = list(np.random.choice(population, size=n_bp, replace=False))
		else:
			normnalized_unfitnesses = [unfitness_relative_offset + (pop.unfitness-min_unfitness) / unfitness_range for pop in population]
			pareto = lambda x,a: a/((1 + x)**(a + 1))
			fitness_pareto_scores = np.array([pareto(2 * nuf, math.sqrt(generation)) for nuf in normnalized_unfitnesses])
			probabilities = fitness_pareto_scores/np.ndarray.sum(fitness_pareto_scores)
			# print(f"Population probabilities: {probabilities}")
			candidate_population = list(np.random.choice(population, size=n_bp, replace=False, p=probabilities))

		candidate_population_set = set(candidate_population)
		population_set = set(population)
		unfit_population = sorted(list(population_set - candidate_population_set), key=lambda pop: pop.unfitness, reverse=False) # the least unfit go first
		
		# 4 ensure there are enough roots
		root_identifiers: set = set([])
		for candidate in candidate_population:
			root_identifiers.add(candidate.root_identifier)
		
		#  - by simply including roots that are unfit because they are roots
		candidate_root_count: int = len(root_identifiers)
		last_find: int = 0 # to be able to pick up from where we left off
		for root_count in range(candidate_root_count, min_roots):
			for i in range(last_find+1, len(unfit_population)):
				pop_under_examination = unfit_population[i]
				pop_root_identifier = pop_under_examination.root_identifier
				if pop_root_identifier not in root_identifiers:
					candidate_population.append(pop_under_examination)
					root_identifiers.add(pop_root_identifier)
					last_find = i
					break
			else:
				print(f"Could not find another root: have {root_count}, want {min_roots}")
				break
		
		# done here, return the new population and the ones chosen for reproduction from among them
		return (population, sorted(candidate_population, key=lambda pop: pop.unfitness))