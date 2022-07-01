from typing import Callable
import tensorflow as tf
import numpy as np

from .binning import make_binned_series, make_timeseries_from_binned_series

def make_data(variant, args):
	samples_per_period: int = args.samples_per_period
		
	################################################
	# generate training data for the given variant #
	################################################
	training_radius: float = args.radius_training * variant.master_period
	training_low: float = 0.0 if args.sampling_sides == "positive" else -training_radius
	training_high: float = 0.0 if args.sampling_sides == "negative" else +training_radius
	training_sample_count: int = samples_per_period*2*args.radius_training

	if args.sampling_method_training == "random":
		variant_train_x = tf.random.uniform(
			shape=(training_sample_count,),
			minval=training_low,
			maxval=training_high
		)
	elif args.sampling_method_training == "linspace":
		variant_train_x = tf.linspace(
			start=-training_low,
			stop=training_high,
			num=training_sample_count,
		)
	else:
		raise Exception(f"Unknown sampling method '{args.sampling_method}'")
	
	variant_train_y = variant.execute(x=variant_train_x)

	############################################
	# generate eval data for the given variant #
	############################################
	# (always remember to consult args.eval_in_training_radius)
	eval_radius: float = args.radius_evaluation * variant.master_period
	eval_radius_offset: float = training_radius if args.eval_in_training_radius else 0.0
	eval_sample_count: int = samples_per_period * (args.radius_evaluation if args.eval_in_training_radius else args.radius_evaluation - args.radius_training)

	if args.sampling_method_evaluation == "random":
		variant_evaluation_x_positive = tf.random.uniform(
			shape=(eval_sample_count,), 
			minval=+eval_radius_offset, 
			maxval=+eval_radius
		)
		variant_evaluation_x_negative = tf.random.uniform(
			shape=(eval_sample_count,), 
			minval=-eval_radius_offset, 
			maxval=-eval_radius
		)
	elif args.sampling_method_evaluation == "linspace":
		variant_evaluation_x_positive = tf.linspace(
			start=eval_radius_offset,
			stop=eval_radius,
			num=eval_sample_count+1, # +1 for the endpoint
		)
		variant_evaluation_x_negative = tf.linspace(
			start=-eval_radius,
			stop=-eval_radius_offset,
			num=eval_sample_count+1, # +1 for the endpoint
		)
	else:
		raise Exception(f"Unknown sampling method '{args.sampling_method}'")
	
	if args.sampling_sides == "both":
		variant_evaluation_x = tf.concat((variant_evaluation_x_negative, variant_evaluation_x_positive), axis=0)
	elif args.sampling_sides == "positive":
		variant_evaluation_x = variant_evaluation_x_positive
	elif args.sampling_sides == "negative":
		variant_evaluation_x = variant_evaluation_x_negative
	variant_evaluation_y = variant.execute(x=variant_evaluation_x)

	# normalize variant_train_y and variant_evaluation_y
	variant_train_y, variant_evaluation_y, y_normalizer = normalize_y(variant_train_y, variant_evaluation_y)

	# add noise if requested
	if args.noise_stddev != 0.0:
		variant_train_y += tf.random.normal(shape=[training_sample_count,], mean=0.0, stddev=args.noise_stddev)

	return variant_train_x, variant_train_y, variant_evaluation_x, variant_evaluation_y, y_normalizer

def normalize_y(train_y, eval_y):
	training_normalizer: float = tf.math.reduce_max(tf.math.abs(train_y))
	evaluation_normalizer: float = tf.math.reduce_max(tf.math.abs(eval_y))
	higher_normalizing_constant: float = tf.math.maximum(training_normalizer, evaluation_normalizer)

	train_y /= higher_normalizing_constant
	eval_y /= higher_normalizing_constant

	return train_y, eval_y, higher_normalizing_constant

def preprocess_data_for_feedforward_setup(x, y, variant, args):
	inputs = x
	expected_outputs = y
	pivots = x

	return inputs, expected_outputs, pivots 

def preprocess_train_data_for_recurrent_setup(x, y, sequence_length: int, bin_range: float, fallback_fn: Callable, timestep_size: int = 1, stride: int = 1, iterations: int = 10):
	# if the model is recurrent, preprocess the data accordingly
	variant_train_x_binned, variant_train_y_binned, train_indices_binned = make_binned_series(x, y, bin_range=bin_range, fn=fallback_fn)
	time_series_x, time_series_y, xses_for_targets\
		= make_timeseries_from_binned_series(
			variant_train_y_binned,
			variant_train_y_binned,
			x_true_source=variant_train_x_binned,
			sequence_length=sequence_length,
			timestep_size = timestep_size,
			stride = stride,
			iterations = iterations
		)
	inputs = time_series_x
	expected_outputs = time_series_y
	pivots = xses_for_targets

	return inputs, expected_outputs, pivots 

def preprocess_eval_data_for_recurrent_setup(x, y, samples_per_bin: int, fallback_fn: Callable):
	# if the model is recurrent, preprocess the data accordingly
	return np.array(x[::samples_per_bin]), np.array(y[::samples_per_bin])