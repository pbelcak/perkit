# tf and tf-related imports
import tensorflow as tf

# base
from .base import GeneticModel

class nFittestModel(GeneticModel):
	def evolve(self, population, population_to_reproduce, generation: int , train_pair: tuple, fitness_pair: tuple, min_roots: int, n_bp: int, unfitness_relative_offset: float, epochs: int):
		train_data, train_y = train_pair[0], train_pair[1]
		fitness_data, fitness_y = fitness_pair[0], fitness_pair[1]
		print(f"Begging the evolutionary cycle, population: {len(population)}, population to reproduce: {len(population_to_reproduce)}")
		
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
			if pop_i % one_tenth < 1:
				print(f"\r - Training population {(pop_i // one_tenth)*10 : .0f}%")
			
			if pop.fits < 1:
				# print(f" -- Fitting pop #{pop_i}")
				es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
				pop.fit(train_data, train_y, epochs=epochs, batch_size=20, verbose=0, callbacks=[es_callback])
				pop.fits += 1
				# print(f" -- Evaluating pop #{pop_i}")
				pop.unfitness = pop.evaluate(
					x=fitness_data, y=fitness_y, batch_size=20, sample_weight=None, steps=None,
					callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
					return_dict=True, verbose=0
				)['mse']
		
		# 3 sort by fitness
		# print(f"Sorting by fitness")
		sorted_population = sorted(population, key=lambda pop: pop.unfitness, reverse=False)
		candidate_population = sorted_population[0:n_bp]
		unfit_population = sorted_population[n_bp:]
		unfit_population_names = [pop.name for pop in unfit_population]
		
		# 4 ensure there are enough roots
		# print(f"First root handling")
		root_identifiers: set = set([])
		for candidate in candidate_population:
			root_identifiers.add(candidate.root_identifier)
		
		#  - by simply including roots that are unfit because they are roots
		# print(f"Second root handling")
		candidate_root_count: int = len(root_identifiers)
		last_root = len(candidate_population)
		for root_count in range(candidate_root_count, min_roots):
			for i in range(last_root, len(sorted_population)):
				pop_root_identifier = sorted_population[i].root_identifier
				if pop_root_identifier not in root_identifiers:
					candidate_population.append(sorted_population[i])
					root_identifiers.add(pop_root_identifier)
					last_root = i+1
					break
			else:
				print(f"Could not find another root: have {root_count}, want {min_roots}")
				break
		
		# done here, return the new population and the ones chosen for reproduction from among them
		return (population, candidate_population)