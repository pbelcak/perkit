# python imports
import os
import dill as pickle
import time
from argparse import Namespace

# the import to get the benchmark generation methods
from ..benchmark.benchmark import *
from ..benchmark.forms import FunctionVariant
from .models import model_map
from .data import make_data, preprocess_data_for_feedforward_setup, preprocess_train_data_for_recurrent_setup, preprocess_eval_data_for_recurrent_setup

# local imports
from .base import EvaluationModel
from .configuration import parse_cmd_arguments

# wandb
import wandb

#pandas
import pandas as pd

# evaluations
evaluations_schema = {
	'Form': str,
	'Variant': str,
	'Training Radius': int,
	'Evaluation Radius': int,
	'Width': int,
	'Optimizer': str,

	'Trial': int,
	'Timestamp (us)': float,
	'Best Epoch': int,
	
	'Evaluation MSE': float,
	'Evaluation MAE': float,
	'Distance-Adjusted MSE': float,
	'Distance-Adjusted MAE': float,
	
	'Least DA-MSE by Periodic Shift': float,
	'Least DA-MAE by Periodic Shift': float,
	'Least DA-MSE by Periodic Speedup': float,
	'Least DA-MAE by Periodic Speedup': float,
	'Least DA-MSE by Periodic Acceleration': float,
	'Least DA-MAE by Periodic Acceleration': float
}
evaluations_df = pd.DataFrame(columns=evaluations_schema.keys()).astype(evaluations_schema)
evaluations_per_model_list = []

project: str = "periodicity"

def main():
	args = parse_cmd_arguments()
	assert args.eval_in_training_radius or args.radius_training != args.radius_evaluation,\
		"If eval_in_training_radius is False, radius_training must NOT be equal to radius_evaluation (if happens to be the case, it will lead to errors in keras.model.predict())"
	assert args.samples_per_period % args.bins_per_period == 0,\
		"samples-per-period must be divisible by bins-per-period"

	with open(args.variants_path, "rb") as variants_file:
		variants = pickle.load(variants_file)
	
	for modelTypeName in args.models[0]:
		assert modelTypeName in model_map, f"Model '{modelTypeName}' not known, use --help to see the choices"
		modelType, modelIsSequenceBased = model_map[modelTypeName]
		assert not ((args.sampling_sides == "both" or args.sampling_sides == "negative") and modelIsSequenceBased),\
			f"Recurrent models (such as '{modelTypeName}', which you requested) can only be evaluated on the positive side of the training domain (use --sampling-sides=positive)"
		assert not (args.sampling_method_evaluation == "random" and modelIsSequenceBased),\
			f"Recurrent models (such as '{modelTypeName}', which you requested) can only be evaluated on linspace-sampled data (use --sampling-method-evaluation=linspace)"

		for optimizer in args.optimizers:
			wandb.init(project="periodicity", group=str(args.job_id), name=modelTypeName + "_" + optimizer, tags=[str(args.job_id), modelTypeName, optimizer], reinit=True)
			
			trials_total = len(variants) * (args.width_max - args.width_min + 1) * args.trials_count
			trials_run = 0
			for variant in variants:
				print(f"Master period: {variant.master_period}")
				for width in range(args.width_min, args.width_max+1, args.width_step):
					for trialNr in range(1, args.trials_count+1):
						result = handle_trial(trialNr, modelType, modelIsSequenceBased, width, variant, optimizer, args)
						evaluations_per_model_list.append(result)

						trials_run += 1
						wandb.log({ 'trial': trials_run, 'progress': (100.0*trials_run / trials_total) })

		evaluations_df = pd.DataFrame.from_records(evaluations_per_model_list)
		evaluations_df.to_csv(
			path_or_buf=os.path.join(args.output_path, str(args.job_id) + "_" + modelTypeName + "_runs.csv"), sep=';',
			header=True, index=False, index_label=None,
			mode='w', line_terminator='\n', decimal='.'
		)
		stats, report_addendum = calculate_stats(evaluations_per_model_list)
		stats_df = pd.DataFrame.from_records(stats)
		stats_df.to_csv(
			path_or_buf=os.path.join(args.output_path, str(args.job_id) + "_" + modelTypeName + '_'.join(args.optimizers) + "_stats.csv"), sep=';',
			header=True, index=False, index_label=None,
			mode='w', line_terminator='\n', decimal='.'
		)

		evaluations_per_model_list.clear()
		
		
def handle_trial(trial_nr: int, modelType: type, modelIsSequenceBased: bool, width: int, variant: FunctionVariant, optimizer, args: Namespace):
	# generate generic data for this trial
	train_x, train_y, eval_x, eval_y, y_normalizer = make_data(variant, args)

	# do model-specific preprocessing
	train_x, train_y, train_pivots, eval_x, eval_y, eval_pivots = preprocess_trial(train_x, train_y, eval_x, eval_y, modelIsSequenceBased, variant, args)

	# make an instance of type modelType
	model = modelType(width=width, args=args)

	# train the model
	best_epoch_n = model.train(
		train_x, train_y, train_pivots, optimizer,
		epochs_max=args.epochs_max,
		validation_x=eval_x,
		validation_y=eval_y,
		max_generations=args.max_generations,
		initial_population_size=args.initial_population_size,
		root_population_size=args.root_population_size,
		children_population_size=args.children_population_size
	)

	# evaluate the model
	epsilon = variant.master_period * args.epsilon_metric_fraction
	evaluation_output = model.evaluate(eval_x, eval_y, eval_pivots, train_y, y_normalizer, variant, args.radius_training, epsilon=epsilon, samples_per_epsilon=args.epsilon_metric_samples)
	
	# add static info to the eval results
	timestamp_us = time.time()
	evaluation_results = {
		'Form': str(variant.form),
		'Variant': str(variant),
		'Training Radius': args.radius_training,
		'Evaluation Radius': args.radius_evaluation,
		'Width': width,

		'Optimizer': optimizer,
		'Trial': trial_nr,
		'Timestamp (us)': timestamp_us,
		'Best Epoch': best_epoch_n,

		**evaluation_output,
	}
	
	return evaluation_results

def preprocess_trial(train_x, train_y, eval_x, eval_y, modelIsSequenceBased: bool, variant: FunctionVariant, args: Namespace):
	if modelIsSequenceBased:
		bin_range: float = variant.master_period / args.bins_per_period
		train_x, train_y, train_pivots = preprocess_train_data_for_recurrent_setup(train_x, train_y, sequence_length=args.sequence_length, bin_range=bin_range, fallback_fn=lambda x: variant.execute(x=np.array(x)))
		
		samples_per_bin: int = int(args.samples_per_period / args.bins_per_period)
		eval_x, eval_y = preprocess_eval_data_for_recurrent_setup(eval_x, eval_y, samples_per_bin, fallback_fn=lambda x: variant.execute(x=np.array(x)))
		eval_pivots = None
	else:
		train_x, train_y, train_pivots = preprocess_data_for_feedforward_setup(train_x, train_y, variant, args)
		eval_x, eval_y, eval_pivots = preprocess_data_for_feedforward_setup(eval_x, eval_y, variant, args)

	return train_x, train_y, train_pivots, eval_x, eval_y, eval_pivots

def calculate_stats(evaluations_per_model_list: list):
	stat_entries = []
	report_entry = {}

	s, r = calculate_stat(evaluations_per_model_list, 'Evaluation MSE')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Evaluation MAE')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Distance-Adjusted MSE')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Distance-Adjusted MAE')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MSE by Periodic Shift')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MAE by Periodic Shift')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MSE by Periodic Speedup')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MAE by Periodic Speedup')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MSE by Periodic Acceleration')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	s, r = calculate_stat(evaluations_per_model_list, 'Least DA-MAE by Periodic Acceleration')
	stat_entries.append(s)
	report_entry = {**report_entry, **r}

	return stat_entries, report_entry

def calculate_stat(evaluations_per_model_list: list, stat_name: str):
	values = []
	for result in evaluations_per_model_list:
		values.append(result[stat_name])

	values = np.array(values)
	
	mean = np.mean(values)
	stddev = np.std(values)
	median = np.median(values)
	quartile_1st = np.quantile(values, 0.25)
	quartile_3rd = np.quantile(values, 0.75)
	iqr = quartile_3rd - quartile_1st

	stat_entry = {
		"stat_name": stat_name,
		"mean": mean,
		"stddev": stddev,
		"median": median,
		"quartile_1st": quartile_1st,
		"quartile_3rd": quartile_3rd,
		"iqr": iqr
	}

	report_entry = {
		stat_name + " mean": mean,
		stat_name + " stddev": stddev,
		stat_name + " median": median,
		stat_name + " quartile_1st": quartile_1st,
		stat_name + " quartile_3rd": quartile_3rd,
		stat_name + " iqr": iqr
	}

	return stat_entry, report_entry

if __name__ == "__main__":
	main()